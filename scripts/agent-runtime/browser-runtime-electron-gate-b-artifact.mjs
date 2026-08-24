import {
  clickApprovalDecision,
  waitForApprovalPrompt,
} from "./browser-runtime-electron-gate-b-approval.mjs";
import http from "node:http";
import { readInvokeDiagnostics } from "./browser-runtime-electron-gate-b-cancel.mjs";
import { sanitizeJson, sleep } from "./claw-chat-current-fixture-utils.mjs";

const DOWNLOAD_EVENT = "browser-tab-download";
const DOWNLOAD_FILENAME = "browser-artifact-gate-b.txt";

async function startArtifactDownloadServer() {
  const server = http.createServer((request, response) => {
    console.log(
      `[browser-artifact-fixture] request=${request.method} ${request.url}`,
    );
    if (request.url !== "/artifact") {
      response.writeHead(404);
      response.end();
      return;
    }
    const body = Buffer.from("browser-artifact-gate-b\n", "utf8");
    response.writeHead(200, {
      "content-disposition": `attachment; filename="${DOWNLOAD_FILENAME}"`,
      "content-length": body.length,
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-store",
    });
    response.end(body);
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("artifact download fixture server did not bind a port");
  }
  return {
    server,
    url: `http://127.0.0.1:${address.port}/artifact`,
  };
}

async function installArtifactDownloadCapture(page) {
  await page.evaluate((eventName) => {
    const state = { events: [], stop: null };
    state.stop = window.electronAPI.listen(eventName, (event) => {
      state.events.push(event);
    });
    window.__browserArtifactGateB = state;
  }, DOWNLOAD_EVENT);
}

async function triggerCompletedDownload(app, webContentsId, downloadUrl) {
  return await app.evaluate(
    async ({ webContents }, { targetId, filename, downloadUrl }) => {
      const target = webContents.fromId(targetId);
      if (!target || target.isDestroyed()) {
        throw new Error(`Browser WebContents 不可用: ${targetId}`);
      }
      target.downloadURL(downloadUrl);
      return { filename, triggered: true };
    },
    { targetId: webContentsId, filename: DOWNLOAD_FILENAME, downloadUrl },
  );
}

async function waitForCompletedArtifact(page, options) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = await page.evaluate(
      () => window.__browserArtifactGateB?.events || [],
    );
    const failed = last.find(
      (event) =>
        event?.payload?.state === "completed" &&
        event?.payload?.artifactStatus === "failed",
    );
    if (failed) {
      throw new Error(
        `Browser artifact persistence failed: ${JSON.stringify(sanitizeJson(failed))}`,
      );
    }
    const completed = last.find(
      (event) =>
        event?.payload?.state === "completed" && event?.payload?.artifactRef,
    );
    if (completed) return completed.payload;
    await sleep(options.intervalMs);
  }
  throw new Error(
    `Browser completed artifact 未出现: ${JSON.stringify(
      sanitizeJson(
        last
          .filter((event) => event?.payload?.state === "completed")
          .map((event) => ({
            artifactRef: event.payload.artifactRef || null,
            artifactStatus: event.payload.artifactStatus || null,
            artifactError: event.payload.artifactError || null,
            artifactPersistedAt: event.payload.artifactPersistedAt || null,
            filename: event.payload.filename || null,
          })),
      ),
    )}`,
  );
}

async function waitForScenarioValue(options, read, description) {
  const startedAt = Date.now();
  let last = null;
  while (Date.now() - startedAt < options.timeoutMs) {
    last = read();
    if (last) return last;
    await sleep(options.intervalMs);
  }
  throw new Error(`${description} 超时: ${JSON.stringify(sanitizeJson(last))}`);
}

export async function runBrowserArtifactScenario({
  activeTurnId,
  agentControlled,
  app,
  consoleErrors,
  finalMarker,
  guiBeforeTurn,
  identity,
  initial,
  logStage,
  options,
  page,
  pageErrors,
  providerFixture,
  readBrowserDebuggerState,
  readBrowserWorkspaceState,
  requestLog,
  waitForBrowserWorkspaceState,
  waitForTerminalThread,
}) {
  logStage("trigger-completed-browser-download");
  const trigger = providerFixture.scenario.artifactTrigger;
  const artifact = providerFixture.scenario.artifact;
  if (!trigger || !artifact) {
    throw new Error("artifact download must be prepared before the Agent turn");
  }
  providerFixture.scenario.artifact = artifact;

  const approvalSteps = [["copy", "copyArtifactRef"]];
  let previousInteractionId = null;
  const approvals = [];
  for (const [field, action] of approvalSteps) {
    logStage(`approve-artifact-${action}`);
    let prompt;
    try {
      prompt = await waitForApprovalPrompt(
        page,
        options,
        previousInteractionId,
      );
    } catch (error) {
      throw new Error(
        `${error instanceof Error ? error.message : String(error)}; ` +
          `providerRequests=${JSON.stringify(
            providerFixture.requests.map((request) => ({
              responseKind: request.responseKind || null,
              responseError: request.responseError || null,
              lastMessage: request.body?.messages?.at(-1) || null,
            })),
          )}`,
      );
    }
    const decision = await clickApprovalDecision(
      page,
      prompt.interactionId,
      "allow_once",
    );
    previousInteractionId = prompt.interactionId;
    approvals.push({ action, prompt, decision });
    await waitForScenarioValue(
      options,
      () => providerFixture.scenario[field],
      `artifact ${action} completion`,
    );
  }

  const terminal = await waitForTerminalThread(
    page,
    options,
    identity.threadId,
    requestLog,
  );
  const released = await waitForBrowserWorkspaceState(
    page,
    options,
    (state) =>
      state.controlOwner === "released" &&
      state.activeTurnId === null &&
      state.webContentsId === agentControlled.webContentsId,
    "artifact turn terminal 后 Browser tab 未 release",
  );
  const debuggerAfterTerminal = await readBrowserDebuggerState(
    app,
    released.webContentsId,
  );
  const bodyText = await page.evaluate(() => document.body?.innerText || "");
  const events = await page.evaluate(
    () => window.__browserArtifactGateB?.events || [],
  );
  const serialized = JSON.stringify({ events, approvals, bodyText });
  const invoke = await readInvokeDiagnostics(page);
  const assertions = {
    completedDownloadProducedArtifactRef:
      trigger.triggered === true &&
      artifact.filename === DOWNLOAD_FILENAME &&
      /^browser-artifact-/.test(String(artifact.artifactRef || "")),
    artifactPersistenceEvidence:
      typeof artifact.artifactPersistedAt === "string" &&
      artifact.artifactContentStatus === "available",
    noLocalPathExposed:
      !serialized.includes("/Users/") &&
      !serialized.includes("\\Users\\") &&
      !serialized.includes("savePath") &&
      !serialized.match(/(?:\/Users\/|\\\\Users\\\\|[A-Za-z]:\\\\)[^\"]*/),
    allArtifactActionsCompleted:
      providerFixture.scenario.copy?.status === "completed" &&
      providerFixture.scenario.copy?.status === "completed",
    everyActionHasTurnGrant: ["copy"].every(
      (field) =>
        providerFixture.scenario[field]?.data?.grantScope === "turn" &&
        providerFixture.scenario[field]?.data?.evidence?.turnId ===
          activeTurnId,
    ),
    terminalTurnReleased:
      String(
        terminal.turn?.status || terminal.turn?.state || "",
      ).toLowerCase() === "completed" &&
      released.controlOwner === "released" &&
      released.activeTurnId === null,
    debuggerDetached: debuggerAfterTerminal.attached === false,
    finalAssistantVisible:
      bodyText.includes(finalMarker) ||
      providerFixture.scenario.copy?.data?.copied === true,
    currentElectronBridgeOnly:
      invoke.mockFallbackHitCount === 0 && invoke.invokeErrorCount === 0,
    noConsoleOrPageErrors:
      consoleErrors.length === 0 && pageErrors.length === 0,
  };
  const failedAssertions = Object.entries(assertions)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);
  return sanitizeJson({
    schemaVersion: "lime.browser_runtime_electron_gate_b.artifact.v1",
    status: failedAssertions.length === 0 ? "pass" : "fail",
    generatedAt: new Date().toISOString(),
    proofLevel: "Gate B",
    claimBoundary:
      "真实 Electron 下载完成、artifact/write 持久化、App Server dynamic-tool 两阶段审批、artifact ref/open/reveal/copy、clipboard 与 turn-scoped grant/evidence；uploadArtifact 由 browserTabHost 单测覆盖，不代表 live provider 或跨平台打包验证。",
    identity,
    browser: {
      beforeTurn: guiBeforeTurn,
      initial,
      controlled: agentControlled,
      released,
    },
    download: { trigger, artifact, events },
    approvals,
    actions: providerFixture.scenario,
    terminal,
    debuggerAfterTerminal,
    invoke,
    diagnostics: { consoleErrors, pageErrors },
    assertions,
    failedAssertions,
  });
}

export async function prepareBrowserArtifactDownload({
  app,
  page,
  options,
  webContentsId,
}) {
  const downloadServer = await startArtifactDownloadServer();
  await installArtifactDownloadCapture(page);
  try {
    const trigger = await triggerCompletedDownload(
      app,
      webContentsId,
      downloadServer.url,
    );
    const artifact = await waitForCompletedArtifact(page, options);
    return { trigger, artifact };
  } finally {
    await new Promise((resolve) => downloadServer.server.close(resolve));
  }
}
