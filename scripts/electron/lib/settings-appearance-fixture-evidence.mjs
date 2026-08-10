import path from "node:path";

export const APPEARANCE_SCENARIO_ID = "appearance-persistence";
export const APPEARANCE_REQUIRED_CONFIG_METHODS = [
  "config/read",
  "config/batchWrite",
];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const LEGACY_APPEARANCE_COMMANDS = [
  "get_appearance_config",
  "save_appearance_config",
  "set_theme",
];
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings Appearance ${label}`);
  }
  return normalized;
}

export function parseSettingsAppearanceFixtureArgs(
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
      `standalone-settings-appearance-${new Date()
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
      "settings-appearance-persistence",
    );
  }
  return options;
}

function parseTrace(raw) {
  try {
    const parsed = JSON.parse(raw || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function appServerMethods(entries) {
  return Array.from(
    new Set(
      entries.flatMap((entry) => {
        const lines = entry?.args_preview?.request?.lines;
        if (!Array.isArray(lines)) return [];
        return lines.flatMap((line) => {
          try {
            const request = JSON.parse(String(line));
            return typeof request?.method === "string" ? [request.method] : [];
          } catch {
            return [];
          }
        });
      }),
    ),
  );
}

export function summarizeSettingsAppearanceTrace(traceRaws) {
  const entries = (Array.isArray(traceRaws) ? traceRaws : [traceRaws]).flatMap(
    parseTrace,
  );
  const appServerEntries = entries.filter(
    (entry) => entry?.command === APP_SERVER_COMMAND,
  );
  const appServerIpcEntries = appServerEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const commands = new Set(entries.map((entry) => entry?.command));
  const configMethods = appServerMethods(appServerIpcEntries);
  return {
    appServerIpcHitCount: appServerIpcEntries.length,
    methods: configMethods,
    configIpcHitCount: appServerIpcEntries.length,
    configMethods,
    missingConfigMethods: APPEARANCE_REQUIRED_CONFIG_METHODS.filter(
      (method) => !configMethods.includes(method),
    ),
    legacyCommands: LEGACY_APPEARANCE_COMMANDS.filter((command) =>
      commands.has(command),
    ),
    mockFallbackHitCount: appServerEntries.filter(
      (entry) => entry.transport !== "electron-ipc",
    ).length,
  };
}

export function createSettingsAppearanceEvidence({
  candidateRunId,
  startedAt,
  prefix,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-appearance-persistence",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary:
      "Real Electron Appearance persistence across both current owners: theme mode in localStorage and recommendation behavior in app config. The fixture changes both, verifies cold-restart readback, restores both, and verifies final cold-restart readback in an isolated user-data directory. Structured evidence records no appearance values or full config.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-appearance-not-completed",
    nextAction:
      "Run the real Electron Appearance fixture through dual-owner save, restart, restore, and final readback.",
    settingsScenarioProof: {
      scenarioId: APPEARANCE_SCENARIO_ID,
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
      screenshot: `${prefix}-saved.png`,
      behaviorScreenshot: `${prefix}-behavior.png`,
      restartScreenshot: `${prefix}-restart.png`,
      restoredScreenshot: `${prefix}-restored.png`,
      rawEvidence: `${prefix}-raw.json`,
      summary: `${prefix}-summary.json`,
    },
  };
}

export function applyPassingSettingsAppearanceEvidence(summary, facts) {
  const { trace, consoleErrors, pageErrors } = facts;
  summary.bridge = {
    electron: facts.electronLaunchCount === 3,
    preloadInvoke: facts.preloadLaunchCount === 3,
    transport: trace.appServerIpcHitCount > 0 ? "electron-ipc" : null,
    command: APP_SERVER_COMMAND,
    appServerIpcHitCount: trace.appServerIpcHitCount,
    methods: trace.methods,
  };
  summary.config = {
    transport: trace.configIpcHitCount > 0 ? "electron-ipc" : null,
    methods: trace.configMethods,
  };
  summary.lifecycle = {
    isolatedUserData: facts.isolatedUserData === true,
    themeChanged: facts.themeChanged === true,
    behaviorChanged: facts.behaviorChanged === true,
    restartThemeReadback: facts.restartThemeReadback === true,
    restartBehaviorReadback: facts.restartBehaviorReadback === true,
    restorationSaved: facts.restorationSaved === true,
    restorationThemeReadback: facts.restorationThemeReadback === true,
    restorationBehaviorReadback: facts.restorationBehaviorReadback === true,
  };
  summary.errors = {
    consoleErrorCount: consoleErrors.length,
    pageErrorCount: pageErrors.length,
    invokeErrorCount: facts.invokeErrorCount,
    legacyCommandHitCount: trace.legacyCommands.length,
    legacyCommands: trace.legacyCommands,
    mockFallbackHitCount: trace.mockFallbackHitCount,
  };
  const checks = [
    ["threeRealElectronLaunches", summary.bridge.electron],
    ["threePreloadInvokeBridges", summary.bridge.preloadInvoke],
    ["appServerElectronIpc", summary.bridge.appServerIpcHitCount > 0],
    ["appServerCurrentMethod", summary.bridge.methods.length > 0],
    ["configElectronIpc", trace.configIpcHitCount > 0],
    ["configCurrentReadWrite", trace.missingConfigMethods.length === 0],
    ["isolatedUserData", summary.lifecycle.isolatedUserData],
    ["appearanceTabActive", facts.appearanceTabActive === true],
    ["themeControlReady", facts.themeControlReady === true],
    ["behaviorControlReady", facts.behaviorControlReady === true],
    ["themeChanged", summary.lifecycle.themeChanged],
    ["behaviorChanged", summary.lifecycle.behaviorChanged],
    ["restartThemeReadback", summary.lifecycle.restartThemeReadback],
    ["restartBehaviorReadback", summary.lifecycle.restartBehaviorReadback],
    ["restorationSaved", summary.lifecycle.restorationSaved],
    ["restorationThemeReadback", summary.lifecycle.restorationThemeReadback],
    [
      "restorationBehaviorReadback",
      summary.lifecycle.restorationBehaviorReadback,
    ],
    ["loadingCleared", facts.loadingVisible === false],
    ["readErrorHidden", facts.errorVisible === false],
    ["consoleErrorsZero", consoleErrors.length === 0],
    ["pageErrorsZero", pageErrors.length === 0],
    ["invokeErrorsZero", facts.invokeErrorCount === 0],
    ["legacyCommandsZero", trace.legacyCommands.length === 0],
    ["mockFallbackZero", trace.mockFallbackHitCount === 0],
    ["savedScreenshotWritten", facts.savedScreenshotWritten === true],
    ["behaviorScreenshotWritten", facts.behaviorScreenshotWritten === true],
    ["restartScreenshotWritten", facts.restartScreenshotWritten === true],
    ["restoredScreenshotWritten", facts.restoredScreenshotWritten === true],
  ];
  const failed = checks.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(
      `SETTINGS Appearance evidence failed: ${failed.join(", ")}`,
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

export function applyFailedSettingsAppearanceEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-appearance-persistence-fixture";
  summary.nextAction =
    "Fix the Appearance current storage/config paths and rerun with the same candidate run-id.";
  summary.error = String(error instanceof Error ? error.message : error).slice(
    0,
    500,
  );
  return summary;
}
