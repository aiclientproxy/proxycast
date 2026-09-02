/* global process */
import path from "node:path";

export const ELECTRON_SMOKE_SCENARIO_ID = "SHELL-01-electron-smoke";
export const ELECTRON_SMOKE_PROOF_LEVEL = "Gate B-F";
export const ELECTRON_SMOKE_CLAIM_BOUNDARY =
  "Real Electron startup, renderer reload, preload/IPC, App Server JSON-RPC, Workbench readiness, Settings route, and three-size responsive layout contract; no provider turn, Thread/Turn/Item identity, live-provider, packaged-app, or Codex Desktop pixel-parity claim.";
export const ELECTRON_SMOKE_LAYOUT_PROOF_LEVEL =
  "electron-responsive-layout-contract";
export const ELECTRON_SMOKE_LAYOUT_CLAIM_BOUNDARY =
  "Real Electron BrowserWindow setSize/getSize and renderer geometry across 1536x960, 1280x800, and 980x680; this is a responsive layout contract, not Codex Desktop pixel-level visual parity or Windows packaged evidence.";

export const ELECTRON_SMOKE_LAYOUT_VIEWPORTS = [
  { width: 1536, height: 960 },
  { width: 1280, height: 800 },
  { width: 980, height: 680 },
] as const;

export interface ElectronSmokeRouteSnapshot {
  stage: "startup" | "workbench" | "workbench-reload" | "settings-memory";
  ready: boolean;
  location: string | null;
  title?: string | null;
}

export interface ElectronSmokeRendererEvidence {
  electron: boolean;
  preloadInvoke: boolean;
  appServerCommandSupported: boolean;
  appServerIpcHitCount: number;
  appServerMethods: string[];
  invokeErrorCount: number;
  traceErrorCount: number;
  legacyCommandHitCount: number;
  legacyCommands: string[];
  mockFallbackHitCount: number;
  pageErrorCount: number;
}

export interface ElectronSmokeLayoutNodeEvidence {
  testId: string;
  present: boolean;
  visible: boolean;
  required: boolean;
  rect: {
    x: number;
    y: number;
    width: number;
    height: number;
    right: number;
    bottom: number;
  } | null;
  withinViewport: boolean;
}

export interface ElectronSmokeLayoutViewportEvidence {
  requested: { width: number; height: number };
  window: { width: number; height: number };
  viewport: { width: number; height: number };
  document: { scrollWidth: number; scrollHeight: number };
  nodes: Record<string, ElectronSmokeLayoutNodeEvidence>;
  assertions: {
    windowSizeMatchesRequest: boolean;
    requiredNodesVisible: boolean;
    requiredNodesWithinViewport: boolean;
    contentSurfacePresent: boolean;
    noHorizontalOverflow: boolean;
    headerTitleVisible: boolean;
    headerActionsDoNotOverlap: boolean;
  };
}

export interface ElectronSmokeLayoutEvidence {
  proofLevel: typeof ELECTRON_SMOKE_LAYOUT_PROOF_LEVEL;
  claimBoundary: typeof ELECTRON_SMOKE_LAYOUT_CLAIM_BOUNDARY;
  viewports: ElectronSmokeLayoutViewportEvidence[];
  screenshots: string[];
  assertions: {
    expectedViewportCount: number;
    capturedViewportCount: number;
    allViewportsPass: boolean;
    composerHeightStable: boolean;
    composerHeightRange: number | null;
  };
}

export interface ElectronSmokeDiagnostics {
  consoleErrorCount: number;
  rendererCrashCount: number;
  rendererUnresponsiveCount: number;
  preloadErrorCount: number;
  rendererLoadErrorCount: number;
}

export interface ElectronSmokeSummaryInput {
  runId: string;
  startedAt: string;
  completedAt: string;
  appVersion: string;
  backendMode: string;
  hostAppServerInitialized: boolean;
  hostAppServerProtocol: string | null;
  routes: ElectronSmokeRouteSnapshot[];
  renderer: ElectronSmokeRendererEvidence;
  layout: ElectronSmokeLayoutEvidence;
  diagnostics: ElectronSmokeDiagnostics;
  artifacts: {
    summary: string;
    trace: string | null;
    screenshot: string | null;
    screenshotCaptured: boolean;
  };
  failureStage?: string | null;
}

export interface ElectronSmokeSummary {
  schemaVersion: 1;
  scenarioId: typeof ELECTRON_SMOKE_SCENARIO_ID;
  priority: "P0";
  proofLevel: typeof ELECTRON_SMOKE_PROOF_LEVEL;
  claimBoundary: typeof ELECTRON_SMOKE_CLAIM_BOUNDARY;
  candidateRunId: string;
  platform: {
    os: string;
    arch: string;
    appVersion: string;
  };
  backendMode: string;
  startedAt: string;
  completedAt: string;
  result: "pass" | "fail";
  failedStage: string | null;
  failureClass: "harness" | "product" | null;
  nextAction: string | null;
  surfaceProof: {
    surfaceId: "SHELL-01";
    proof: "gate-b-f";
    complete: boolean;
  };
  routes: ElectronSmokeRouteSnapshot[];
  layout: ElectronSmokeLayoutEvidence;
  bridge: {
    electron: boolean;
    preloadInvoke: boolean;
    transport: "electron-ipc" | null;
    command: "app_server_handle_json_lines";
    appServerIpcHitCount: number;
    methods: string[];
    hostInitialized: boolean;
    hostProtocol: string | null;
  };
  errors: ElectronSmokeDiagnostics & {
    invokeErrorCount: number;
    traceErrorCount: number;
    pageErrorCount: number;
    legacyCommandHitCount: number;
    legacyCommands: string[];
    mockFallbackHitCount: number;
  };
  assertions: {
    total: number;
    passed: number;
    failed: string[];
    details: Record<string, boolean>;
  };
  artifacts: {
    summary: string;
    trace: string | null;
    screenshot: string | null;
  };
}

export function normalizeElectronSmokeRunId(
  value: string | undefined,
  fallback: string,
): string {
  const runId = value?.trim() || fallback.trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(runId)) {
    throw new Error(
      "LIME_GATE_RUN_ID must contain only letters, digits, dots, underscores, and hyphens, with a maximum length of 128",
    );
  }
  return runId;
}

export function isElectronSmokeStartupUrl(value: string): boolean {
  if (value.startsWith("data:text/html")) {
    return true;
  }
  try {
    return (
      path.basename(new URL(value).pathname) === "main-window-startup.html"
    );
  } catch {
    return false;
  }
}

export function sanitizeElectronSmokeLocation(
  value: string | undefined,
): string | null {
  if (!value) {
    return null;
  }
  try {
    const url = new URL(value);
    const queryKeys = [...url.searchParams.keys()].sort();
    const query = queryKeys.length > 0 ? `?${queryKeys.join("&")}` : "";
    if (url.protocol === "file:") {
      return `file:///${path.basename(url.pathname)}${query}`;
    }
    return `${url.origin}${url.pathname}${query}`;
  } catch {
    return null;
  }
}

export function buildElectronSmokeSummary(
  input: ElectronSmokeSummaryInput,
): ElectronSmokeSummary {
  const routeStages = new Map(
    input.routes.map((route) => [route.stage, route]),
  );
  const assertionDetails = {
    startupVisible: routeStages.get("startup")?.ready === true,
    workbenchReady: routeStages.get("workbench")?.ready === true,
    workbenchReloadReady: routeStages.get("workbench-reload")?.ready === true,
    settingsMemoryReady: routeStages.get("settings-memory")?.ready === true,
    electronRenderer: input.renderer.electron,
    electronPreloadInvokeAvailable: input.renderer.preloadInvoke,
    appServerCommandSupported: input.renderer.appServerCommandSupported,
    electronIpcAppServerBridgeUsed: input.renderer.appServerIpcHitCount > 0,
    currentAppServerMethodObserved: input.renderer.appServerMethods.length > 0,
    hostAppServerInitialized: input.hostAppServerInitialized,
    noConsoleErrors: input.diagnostics.consoleErrorCount === 0,
    noPageErrors: input.renderer.pageErrorCount === 0,
    noInvokeErrors:
      input.renderer.invokeErrorCount === 0 &&
      input.renderer.traceErrorCount === 0,
    noRendererCrashes: input.diagnostics.rendererCrashCount === 0,
    noRendererUnresponsiveEvents:
      input.diagnostics.rendererUnresponsiveCount === 0,
    noPreloadErrors: input.diagnostics.preloadErrorCount === 0,
    noRendererLoadErrors: input.diagnostics.rendererLoadErrorCount === 0,
    noLegacyCommandHits: input.renderer.legacyCommandHitCount === 0,
    noMockFallbackHits: input.renderer.mockFallbackHitCount === 0,
    traceCaptured: input.artifacts.trace !== null,
    screenshotCaptured: input.artifacts.screenshotCaptured,
    layoutViewportsCaptured:
      input.layout.viewports.length === ELECTRON_SMOKE_LAYOUT_VIEWPORTS.length,
    layoutGeometryStable:
      input.layout.assertions.allViewportsPass &&
      input.layout.assertions.composerHeightStable,
    layoutScreenshotsCaptured:
      input.layout.screenshots.length ===
      ELECTRON_SMOKE_LAYOUT_VIEWPORTS.length,
  };
  const failed = Object.entries(assertionDetails)
    .filter(([, passed]) => !passed)
    .map(([name]) => name);

  const result = failed.length === 0 && !input.failureStage ? "pass" : "fail";
  const failedStage =
    input.failureStage ?? (failed.length > 0 ? "contract-assertions" : null);

  return {
    schemaVersion: 1,
    scenarioId: ELECTRON_SMOKE_SCENARIO_ID,
    priority: "P0",
    proofLevel: ELECTRON_SMOKE_PROOF_LEVEL,
    claimBoundary: ELECTRON_SMOKE_CLAIM_BOUNDARY,
    candidateRunId: input.runId,
    platform: {
      os: process.platform,
      arch: process.arch,
      appVersion: input.appVersion,
    },
    backendMode: input.backendMode,
    startedAt: input.startedAt,
    completedAt: input.completedAt,
    result,
    failedStage,
    failureClass:
      result === "fail" ? classifyElectronSmokeFailure(failed) : null,
    nextAction:
      result === "fail"
        ? "Inspect the failed SHELL-01 assertions, fix the owning product or harness boundary, and rerun on the same candidate."
        : null,
    surfaceProof: {
      surfaceId: "SHELL-01",
      proof: "gate-b-f",
      complete: result === "pass",
    },
    routes: input.routes,
    layout: input.layout,
    bridge: {
      electron: input.renderer.electron,
      preloadInvoke: input.renderer.preloadInvoke,
      transport:
        input.renderer.appServerIpcHitCount > 0 ? "electron-ipc" : null,
      command: "app_server_handle_json_lines",
      appServerIpcHitCount: input.renderer.appServerIpcHitCount,
      methods: [...new Set(input.renderer.appServerMethods)].sort(),
      hostInitialized: input.hostAppServerInitialized,
      hostProtocol: input.hostAppServerProtocol,
    },
    errors: {
      ...input.diagnostics,
      invokeErrorCount: input.renderer.invokeErrorCount,
      traceErrorCount: input.renderer.traceErrorCount,
      pageErrorCount: input.renderer.pageErrorCount,
      legacyCommandHitCount: input.renderer.legacyCommandHitCount,
      legacyCommands: [...new Set(input.renderer.legacyCommands)].sort(),
      mockFallbackHitCount: input.renderer.mockFallbackHitCount,
    },
    assertions: {
      total: Object.keys(assertionDetails).length,
      passed: Object.keys(assertionDetails).length - failed.length,
      failed,
      details: assertionDetails,
    },
    artifacts: {
      summary: input.artifacts.summary,
      trace: input.artifacts.trace,
      screenshot: input.artifacts.screenshot,
    },
  };
}

function classifyElectronSmokeFailure(
  failedAssertions: string[],
): "harness" | "product" {
  const harnessOnly = new Set([
    "startupVisible",
    "traceCaptured",
    "screenshotCaptured",
  ]);
  return failedAssertions.length > 0 &&
    failedAssertions.every((assertion) => harnessOnly.has(assertion))
    ? "harness"
    : "product";
}
