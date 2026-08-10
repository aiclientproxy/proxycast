import fs from "node:fs";
import path from "node:path";

import { collectTraceRequestMethods } from "./claw-chat-current-fixture-rpc.mjs";
import { buildGateBExecutionEvidence } from "./claw-chat-current-fixture-gate-b-execution-evidence.mjs";

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const RETIRED_COMMANDS = loadRetiredCommands();

export function buildGateBContractEvidence({
  traceMessages,
  rendererSnapshot,
  pageErrors,
  pageLifecycleEvents,
  runId,
  artifacts,
  appServerRequests,
  backendLedger,
  guiEvidence,
  expectedIdentity,
}) {
  const traces = Array.isArray(traceMessages) ? traceMessages : [];
  const appServerIpcEntries = traces.filter(
    (entry) =>
      entry?.command === APP_SERVER_COMMAND &&
      entry?.transport === "electron-ipc" &&
      entry?.status === "success",
  );
  const successfulAppServerRequests = (
    Array.isArray(appServerRequests) ? appServerRequests : []
  ).filter(
    (entry) =>
      typeof entry?.method === "string" &&
      entry.method.trim().length > 0 &&
      !entry?.error,
  );
  const legacyCommandHits = traces
    .map((entry) => entry?.command)
    .filter((command) => RETIRED_COMMANDS.has(command));
  const mockFallbackHits = traces.filter(isMockFallbackTrace).map((entry) => ({
    command: entry?.command ?? null,
    transport: entry?.transport ?? null,
    status: entry?.status ?? null,
  }));
  const crashes = (
    Array.isArray(pageLifecycleEvents) ? pageLifecycleEvents : []
  ).filter((entry) => String(entry?.type || "").includes("crash"));
  const artifactEntries = Object.entries(artifacts ?? {}).filter(
    ([, value]) => typeof value === "string" && value.trim().length > 0,
  );
  const artifactNames = new Set(artifactEntries.map(([name]) => name));
  const artifactDirectories = new Set(
    artifactEntries.map(([, value]) => path.dirname(path.resolve(value))),
  );
  const execution = buildGateBExecutionEvidence({
    traceMessages,
    appServerRequests,
    backendLedger,
    guiEvidence,
    expectedIdentity,
  });

  return {
    run: {
      id: typeof runId === "string" ? runId : null,
      artifactNames: artifactEntries.map(([name, value]) => ({
        name,
        file: path.basename(value),
      })),
      requiredArtifactsPresent: [
        "summary",
        "backendLedger",
        "screenshot",
      ].every((name) => artifactNames.has(name)),
      artifactsShareDirectory:
        artifactEntries.length > 0 && artifactDirectories.size === 1,
      screenshotCaptured: artifactEntries.some(
        ([name, value]) => name === "screenshot" && fs.existsSync(value),
      ),
    },
    renderer: {
      electron: rendererSnapshot?.electron === true,
      preloadInvoke: rendererSnapshot?.hasInvokeBridge === true,
      appServerCommandSupported: rendererSnapshot?.supportsAppServer === true,
      url: rendererSnapshot?.url ?? null,
    },
    appServerIpcHitCount: appServerIpcEntries.length,
    appServerIpcMethods: collectTraceRequestMethods(appServerIpcEntries),
    appServerRequestCount: successfulAppServerRequests.length,
    appServerRequestMethods: [
      ...new Set(successfulAppServerRequests.map((entry) => entry.method)),
    ].sort(),
    legacyCommandHitCount: legacyCommandHits.length,
    legacyCommands: [...new Set(legacyCommandHits)].sort(),
    mockFallbackHitCount: mockFallbackHits.length,
    mockFallbackHits,
    pageErrorCount: Array.isArray(pageErrors) ? pageErrors.length : 0,
    pageCrashCount: crashes.length,
    identity: execution.identity,
    outcome: execution.outcome,
    collaborationMode: execution.collaborationMode,
    permissionProfile: execution.permissionProfile,
  };
}

export function buildGateBContractAssertions(evidence) {
  const assertions = {
    runIdPresent:
      typeof evidence.run.id === "string" && evidence.run.id.trim().length > 0,
    screenshotCaptured: evidence.run.screenshotCaptured === true,
    evidenceArtifactsComplete: evidence.run.requiredArtifactsPresent === true,
    evidenceArtifactsShareRunDirectory:
      evidence.run.artifactsShareDirectory === true,
    electronRenderer: evidence.renderer.electron === true,
    electronPreloadInvokeAvailable: evidence.renderer.preloadInvoke === true,
    appServerCommandSupported:
      evidence.renderer.appServerCommandSupported === true,
    electronIpcAppServerBridgeUsed: evidence.appServerIpcHitCount > 0,
    noLegacyCommandHits: evidence.legacyCommandHitCount === 0,
    noMockFallbackHits: evidence.mockFallbackHitCount === 0,
    noPageErrors: evidence.pageErrorCount === 0,
    noPageCrashes: evidence.pageCrashCount === 0,
    identityConsistent: evidence.identity.consistent === true,
    explicitTerminalOrPending: evidence.outcome.explicit === true,
  };
  if (evidence.permissionProfile.runtime.matchedPrimaryTurn !== true) {
    return assertions;
  }
  return {
    ...assertions,
    permissionProfileCatalogRequested:
      evidence.permissionProfile.catalogRequestCount > 0,
    permissionProfileResolvedBeforeTurnStart:
      evidence.permissionProfile.catalogBeforeTurnStart === true,
    permissionProfileAppliedOnDesktopWire:
      typeof evidence.permissionProfile.wire.profileId === "string" &&
      evidence.permissionProfile.wire.legacySandboxPolicyPresent === false,
    permissionProfileReachedRuntime:
      typeof evidence.permissionProfile.runtime.sandboxPolicy === "string" &&
      evidence.permissionProfile.runtime.activePermissionProfileId ===
        evidence.permissionProfile.wire.profileId,
    permissionProfileWireRuntimeConsistent:
      evidence.permissionProfile.wireRuntimeConsistent === true,
  };
}

function isMockFallbackTrace(entry) {
  if (entry?.mock === true || entry?.mockFallback === true) {
    return true;
  }
  return [
    entry?.transport,
    entry?.source,
    entry?.fallback,
    entry?.fallbackMode,
  ].some(
    (value) =>
      typeof value === "string" && value.toLowerCase().includes("mock"),
  );
}

function loadRetiredCommands() {
  const catalogPath = path.resolve(
    process.cwd(),
    "src/lib/governance/legacySurfaceCatalog.json",
  );
  const catalog = JSON.parse(fs.readFileSync(catalogPath, "utf8"));
  return new Set(
    (Array.isArray(catalog.commands) ? catalog.commands : [])
      .filter((entry) =>
        ["dead", "dead-candidate"].includes(entry?.classification),
      )
      .flatMap((entry) =>
        Array.isArray(entry.commands) ? entry.commands : [],
      ),
  );
}
