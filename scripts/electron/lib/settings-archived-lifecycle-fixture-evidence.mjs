import path from "node:path";

export const ARCHIVED_LIFECYCLE_SCENARIO_ID = "archived-lifecycle-recovery";
export const ARCHIVED_LIFECYCLE_REQUIRED_METHODS = [
  "thread/archive",
  "thread/unarchive",
  "thread/list",
  "thread/read",
];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const LEGACY_COMMAND_PATTERN = /^(?:agent_session_|agent_runtime_)/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings archived lifecycle ${label}`);
  }
  return normalized;
}

export function parseSettingsArchivedLifecycleArgs(
  argv,
  { defaults, cwd = process.cwd() } = {},
) {
  const options = { ...defaults, help: false };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--run-id" && next) {
      options.runId = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--evidence-dir" && next) {
      options.evidenceDir = path.resolve(next.trim());
      index += 1;
      continue;
    }
    if (arg === "--prefix" && next) {
      options.prefix = next.trim();
      index += 1;
      continue;
    }
    if (arg === "--timeout-ms" && next) {
      options.timeoutMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--interval-ms" && next) {
      options.intervalMs = Number(next);
      index += 1;
      continue;
    }
    if (arg === "--keep-temp") {
      options.keepTemp = true;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  if (options.help) return options;
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  options.runId = validateName(
    options.runId ||
      `standalone-settings-archived-${new Date()
        .toISOString()
        .replace(/[-:.]/g, "")}-${process.pid}`,
    "run-id",
  );
  options.prefix = validateName(options.prefix, "prefix");
  if (!options.evidenceDir) {
    options.evidenceDir = path.join(
      cwd,
      ".lime",
      "qc",
      "project-gates",
      options.runId,
      "settings-archived-lifecycle-recovery",
    );
  }
  return options;
}

export function summarizeSettingsArchivedLifecycleTrace(rawEvidence) {
  const guiRequests = [
    rawEvidence?.sidebarGuiArchive?.requests,
    rawEvidence?.settingsGuiRestoreArchive?.requests,
    rawEvidence?.settingsGuiRestore?.requests,
  ].flatMap((requests) => (Array.isArray(requests) ? requests : []));
  const appServerRequests = guiRequests.filter(
    (request) => request?.command === APP_SERVER_COMMAND,
  );
  const appServerIpcRequests = appServerRequests.filter(
    (request) => request?.transport === "electron-ipc",
  );
  const ownerPhaseMethods = [
    rawEvidence?.persistedArchive?.requests,
    rawEvidence?.persistedArchiveReadback?.requests,
    rawEvidence?.persistedUnarchive?.requests,
    rawEvidence?.persistedUnarchiveReadback?.requests,
  ]
    .flatMap((requests) => (Array.isArray(requests) ? requests : []))
    .map((request) => request?.method)
    .filter((method) => typeof method === "string");
  const methods = Array.from(
    new Set([
      ...appServerIpcRequests.map((request) => request.method),
      ...ownerPhaseMethods,
    ]),
  ).sort();
  const legacyCommands = Array.from(
    new Set(
      guiRequests
        .map((request) => String(request?.command ?? ""))
        .filter((command) => LEGACY_COMMAND_PATTERN.test(command)),
    ),
  ).sort();
  return {
    appServerIpcHitCount: appServerIpcRequests.length,
    methods,
    missingMethods: ARCHIVED_LIFECYCLE_REQUIRED_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    invokeErrorCount: appServerIpcRequests.filter(
      (request) => request?.status !== "success",
    ).length,
    legacyCommands,
    mockFallbackHitCount: appServerRequests.filter(
      (request) => request?.transport !== "electron-ipc",
    ).length,
  };
}

export function createSettingsArchivedLifecycleEvidence({
  candidateRunId,
  startedAt,
  prefix,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-archived-lifecycle-recovery",
    priority: "P0",
    proofLevel: "Gate B-R",
    claimBoundary:
      "Real Electron persisted conversation archive from the sidebar, Settings archived-conversation restore, and cold-restart readback through current thread/list/read/update. The owner fixture uses an unavailable model backend, and this adapter stores only lifecycle booleans, current methods, error counts, and screenshots; it does not store conversation content, identities, database rows, paths, or import payloads.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-archived-lifecycle-not-completed",
    nextAction: "Run the real Electron archived lifecycle fixture.",
    settingsScenarioProof: {
      scenarioId: ARCHIVED_LIFECYCLE_SCENARIO_ID,
      complete: false,
    },
    assertions: {
      total: 1,
      passed: 0,
      failed: ["notCompleted"],
      details: {},
    },
    bridge: {
      electron: false,
      preloadInvoke: false,
      transport: null,
      command: APP_SERVER_COMMAND,
      appServerIpcHitCount: 0,
      methods: [],
    },
    errors: {
      consoleErrorCount: 0,
      pageErrorCount: 0,
      invokeErrorCount: 0,
      legacyCommandHitCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
    },
    artifacts: {
      archivedScreenshot: `${prefix}-archived.png`,
      recoveredScreenshot: `${prefix}-recovered.png`,
      finalScreenshot: `${prefix}-recovered.png`,
      screenshot: `${prefix}-recovered.png`,
      rawEvidence: `${prefix}-raw.json`,
      summary: `${prefix}-summary.json`,
    },
  };
}

export function applyPassingSettingsArchivedLifecycleEvidence(summary, facts) {
  const source = facts.sourceSummary;
  const trace = facts.trace;
  const consoleErrors = Array.isArray(source?.consoleErrors)
    ? source.consoleErrors
    : null;
  const pageErrors = Array.isArray(source?.pageErrors)
    ? source.pageErrors
    : null;
  summary.bridge = {
    electron: source?.electronPreloadBridge === true,
    preloadInvoke: source?.electronPreloadBridge === true,
    transport: trace.appServerIpcHitCount > 0 ? "electron-ipc" : null,
    command: APP_SERVER_COMMAND,
    appServerIpcHitCount: trace.appServerIpcHitCount,
    methods: trace.methods,
  };
  summary.lifecycle = {
    ownerFixturePassed: source?.ok === true,
    archivePersisted: Boolean(
      source?.persistedArchiveSummary?.archiveRequestSeen &&
      source?.persistedArchiveSummary?.archivedAfterSession,
    ),
    archiveRestartReadback: Boolean(
      source?.persistedArchiveReopenSummary?.archivedAfterRestartSession,
    ),
    sidebarGuiArchive:
      source?.sidebarGuiArchiveSummary?.updateRequestSeen === true,
    settingsGuiRestore:
      source?.settingsGuiRestoreSummary?.updateRequestSeen === true,
    unarchivePersisted: Boolean(
      source?.persistedUnarchiveSummary?.unarchiveRequestSeen &&
      source?.persistedUnarchiveSummary?.recentAfterSession,
    ),
    unarchiveRestartReadback: Boolean(
      source?.persistedUnarchiveReopenSummary?.recentAfterRestartSession,
    ),
    sidecarRestartReadback: source?.sidecarRestartReadback === true,
  };
  summary.errors = {
    consoleErrorCount: consoleErrors?.length ?? -1,
    pageErrorCount: pageErrors?.length ?? -1,
    invokeErrorCount: trace.invokeErrorCount,
    legacyCommandHitCount: trace.legacyCommands.length,
    legacyCommands: trace.legacyCommands,
    mockFallbackHitCount: trace.mockFallbackHitCount,
  };
  const checks = [
    ["ownerFixturePassed", summary.lifecycle.ownerFixturePassed],
    [
      "realElectronPreload",
      summary.bridge.electron && summary.bridge.preloadInvoke,
    ],
    ["appServerElectronIpc", summary.bridge.appServerIpcHitCount > 0],
    ["currentMethodsComplete", trace.missingMethods.length === 0],
    ["archivePersisted", summary.lifecycle.archivePersisted],
    ["archiveRestartReadback", summary.lifecycle.archiveRestartReadback],
    ["sidebarGuiArchive", summary.lifecycle.sidebarGuiArchive],
    ["settingsGuiRestore", summary.lifecycle.settingsGuiRestore],
    ["unarchivePersisted", summary.lifecycle.unarchivePersisted],
    ["unarchiveRestartReadback", summary.lifecycle.unarchiveRestartReadback],
    ["sidecarRestartReadback", summary.lifecycle.sidecarRestartReadback],
    ["consoleErrorsObservedAndZero", consoleErrors?.length === 0],
    ["pageErrorsObservedAndZero", pageErrors?.length === 0],
    ["invokeErrorsZero", trace.invokeErrorCount === 0],
    ["legacyCommandsZero", trace.legacyCommands.length === 0],
    ["mockFallbackZero", trace.mockFallbackHitCount === 0],
    ["archivedScreenshotWritten", facts.archivedScreenshotWritten === true],
    ["recoveredScreenshotWritten", facts.recoveredScreenshotWritten === true],
  ];
  const failed = checks.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(
      `SETTINGS archived lifecycle evidence failed: ${failed.join(", ")}`,
    );
  }
  summary.result = "pass";
  summary.completedAt = facts.completedAt;
  summary.settingsScenarioProof.complete = true;
  summary.assertions = {
    total: checks.length,
    passed: checks.length,
    failed: [],
    details: Object.fromEntries(checks),
  };
  delete summary.failureClass;
  delete summary.nextAction;
  return summary;
}

export function applyFailedSettingsArchivedLifecycleEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-archived-lifecycle-fixture";
  summary.nextAction =
    "Fix the archived lifecycle current evidence and rerun with a new prefix.";
  summary.errorClass =
    error instanceof Error && error.name ? error.name : "Error";
  return summary;
}
