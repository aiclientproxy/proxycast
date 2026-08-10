import path from "node:path";

export const MEDIA_SERVICES_SCENARIO_ID = "media-services-readiness";
export const MEDIA_SERVICES_REQUIRED_METHODS = [
  "model/list",
  "modelPreferences/list",
  "modelSyncState/read",
];
export const MEDIA_SERVICES_REQUIRED_HOST_COMMANDS = [
  "voice_models_list_catalog",
];
export const MEDIA_SERVICES_REQUIRED_CONFIG_METHODS = ["config/read"];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const LEGACY_MEDIA_SERVICES_COMMANDS = [
  "get_api_key_providers",
  "get_model_registry",
  "list_voice_models",
  "get_voice_input_config",
];
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings Media Services ${label}`);
  }
  return normalized;
}

export function parseSettingsMediaServicesFixtureArgs(
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
      `standalone-settings-media-services-${new Date()
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
      "settings-media-services-readiness",
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

export function summarizeSettingsMediaServicesTrace(traceRaw) {
  const entries = parseTrace(traceRaw);
  const appServerEntries = entries.filter(
    (entry) => entry?.command === APP_SERVER_COMMAND,
  );
  const appServerIpcEntries = appServerEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const hostEntries = entries.filter((entry) =>
    MEDIA_SERVICES_REQUIRED_HOST_COMMANDS.includes(entry?.command),
  );
  const hostIpcEntries = hostEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const methods = appServerMethods(appServerIpcEntries);
  const hostCommands = Array.from(
    new Set(hostIpcEntries.map((entry) => entry.command)),
  );
  const commands = new Set(entries.map((entry) => entry?.command));
  return {
    appServerIpcHitCount: appServerIpcEntries.length,
    methods,
    missingMethods: MEDIA_SERVICES_REQUIRED_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    hostIpcHitCount: hostIpcEntries.length,
    hostCommands,
    configMethods: methods,
    missingConfigMethods: MEDIA_SERVICES_REQUIRED_CONFIG_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    missingHostCommands: MEDIA_SERVICES_REQUIRED_HOST_COMMANDS.filter(
      (command) => !hostCommands.includes(command),
    ),
    legacyCommands: LEGACY_MEDIA_SERVICES_COMMANDS.filter((command) =>
      commands.has(command),
    ),
    mockFallbackHitCount: [...appServerEntries, ...hostEntries].filter(
      (entry) => entry.transport !== "electron-ipc",
    ).length,
  };
}

export function createSettingsMediaServicesEvidence({
  candidateRunId,
  startedAt,
  prefix,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-media-services-readiness",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary:
      "Real Electron Media Services current config, model selector reads, local voice catalog, and terminal GUI readiness. This does not claim a live Provider generation request.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-media-services-not-completed",
    nextAction:
      "Run the real Electron Media Services fixture to a terminal readiness state.",
    settingsScenarioProof: {
      scenarioId: MEDIA_SERVICES_SCENARIO_ID,
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
      screenshot: `${prefix}.png`,
      imageScreenshot: `${prefix}-image.png`,
      videoScreenshot: `${prefix}-video.png`,
      readinessScreenshot: `${prefix}-readiness.png`,
      rawEvidence: `${prefix}-raw.json`,
      summary: `${prefix}-summary.json`,
    },
  };
}

export function applyPassingSettingsMediaServicesEvidence(summary, facts) {
  const { trace, consoleErrors, pageErrors } = facts;
  summary.bridge = {
    electron: facts.electronRenderer === true,
    preloadInvoke: facts.preloadInvoke === true,
    transport: trace.appServerIpcHitCount > 0 ? "electron-ipc" : null,
    command: APP_SERVER_COMMAND,
    appServerIpcHitCount: trace.appServerIpcHitCount,
    methods: trace.methods,
  };
  summary.host = {
    transport: trace.hostIpcHitCount > 0 ? "electron-ipc" : null,
    commands: trace.hostCommands,
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
    ["realElectronRenderer", summary.bridge.electron],
    ["preloadInvokeBridge", summary.bridge.preloadInvoke],
    ["appServerElectronIpc", summary.bridge.appServerIpcHitCount > 0],
    ["modelSelectorCurrentMethods", trace.missingMethods.length === 0],
    ["hostElectronIpc", trace.hostIpcHitCount > 0],
    ["hostCurrentReads", trace.missingHostCommands.length === 0],
    ["mediaServicesTabActive", facts.mediaServicesActive === true],
    ["serviceModelsVisible", facts.serviceModelsVisible === true],
    ["imageServiceVisible", facts.imageServiceVisible === true],
    ["videoServiceVisible", facts.videoServiceVisible === true],
    ["voiceServiceVisible", facts.voiceServiceVisible === true],
    ["configControlsReady", facts.configControlsReady === true],
    ["loadingCleared", facts.loadingVisible === false],
    ["readErrorHidden", facts.errorVisible === false],
    ["consoleErrorsZero", consoleErrors.length === 0],
    ["pageErrorsZero", pageErrors.length === 0],
    ["invokeErrorsZero", facts.invokeErrorCount === 0],
    ["legacyCommandsZero", trace.legacyCommands.length === 0],
    ["mockFallbackZero", trace.mockFallbackHitCount === 0],
    ["screenshotWritten", facts.screenshotWritten === true],
    ["imageScreenshotWritten", facts.imageScreenshotWritten === true],
    ["videoScreenshotWritten", facts.videoScreenshotWritten === true],
    ["readinessScreenshotWritten", facts.readinessScreenshotWritten === true],
  ];
  const failed = checks.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(
      `SETTINGS Media Services evidence failed: ${failed.join(", ")}`,
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

export function applyFailedSettingsMediaServicesEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-media-services-readiness-fixture";
  summary.nextAction =
    "Fix the Media Services current read path and rerun with the same candidate run-id.";
  summary.error = String(error instanceof Error ? error.message : error).slice(
    0,
    500,
  );
  return summary;
}
