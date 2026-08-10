import path from "node:path";

export const ENVIRONMENT_SCENARIO_ID = "environment-current-read";
export const ENVIRONMENT_REQUIRED_CONFIG_METHODS = ["config/read"];
export const ENVIRONMENT_REQUIRED_HOST_COMMANDS = ["get_environment_preview"];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const LEGACY_ENVIRONMENT_COMMANDS = [
  "get_env_vars",
  "get_shell_environment",
  "get_environment_variables",
];
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings Environment ${label}`);
  }
  return normalized;
}

export function parseSettingsEnvironmentFixtureArgs(
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
  if (options.help) {
    return options;
  }
  if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 30_000) {
    throw new Error("--timeout-ms must be >= 30000");
  }
  if (!Number.isFinite(options.intervalMs) || options.intervalMs < 100) {
    throw new Error("--interval-ms must be >= 100");
  }
  options.runId = validateName(
    options.runId ||
      `standalone-settings-environment-${new Date()
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
      "settings-environment-current-read",
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

export function summarizeSettingsEnvironmentTrace(traceRaw) {
  const entries = parseTrace(traceRaw);
  const appServerEntries = entries.filter(
    (entry) => entry?.command === APP_SERVER_COMMAND,
  );
  const appServerIpcEntries = appServerEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const hostEntries = entries.filter((entry) =>
    ENVIRONMENT_REQUIRED_HOST_COMMANDS.includes(entry?.command),
  );
  const commands = new Set(entries.map((entry) => entry?.command));
  const methods = appServerMethods(appServerIpcEntries);
  return {
    appServerIpcHitCount: appServerIpcEntries.length,
    appServerMethods: methods,
    missingConfigMethods: ENVIRONMENT_REQUIRED_CONFIG_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    hostIpcHitCount: hostEntries.filter(
      (entry) => entry.transport === "electron-ipc",
    ).length,
    hostCommands: Array.from(
      new Set(hostEntries.map((entry) => entry.command)),
    ),
    missingHostCommands: ENVIRONMENT_REQUIRED_HOST_COMMANDS.filter(
      (command) => !commands.has(command),
    ),
    legacyCommands: LEGACY_ENVIRONMENT_COMMANDS.filter((command) =>
      commands.has(command),
    ),
    mockFallbackHitCount: [...appServerEntries, ...hostEntries].filter(
      (entry) => entry.transport !== "electron-ipc",
    ).length,
  };
}

export function createSettingsEnvironmentEvidence({
  candidateRunId,
  startedAt,
  prefix,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-environment-current-read",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary:
      "Real Electron Environment Settings config and merged preview reads. Evidence records no environment variable names or values and does not claim save behavior.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-environment-not-completed",
    nextAction:
      "Run the real Electron Environment Settings fixture to a terminal read state.",
    settingsScenarioProof: {
      scenarioId: ENVIRONMENT_SCENARIO_ID,
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
      rawEvidence: `${prefix}-raw.json`,
      summary: `${prefix}-summary.json`,
    },
  };
}

export function applyPassingSettingsEnvironmentEvidence(summary, facts) {
  const { trace, consoleErrors, pageErrors } = facts;
  summary.bridge = {
    electron: facts.electronRenderer === true,
    preloadInvoke: facts.preloadInvoke === true,
    transport: trace.appServerIpcHitCount > 0 ? "electron-ipc" : null,
    command: APP_SERVER_COMMAND,
    appServerIpcHitCount: trace.appServerIpcHitCount,
    methods: trace.appServerMethods,
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
    ["appServerCurrentMethod", summary.bridge.methods.length > 0],
    ["hostElectronIpc", trace.hostIpcHitCount > 0],
    ["hostCurrentCommands", trace.missingHostCommands.length === 0],
    ["configShapeValid", facts.configShapeValid === true],
    ["environmentTabActive", facts.environmentActive === true],
    ["loadingCleared", facts.loadingVisible === false],
    ["readErrorHidden", facts.errorVisible === false],
    ["consoleErrorsZero", consoleErrors.length === 0],
    ["pageErrorsZero", pageErrors.length === 0],
    ["invokeErrorsZero", facts.invokeErrorCount === 0],
    ["legacyCommandsZero", trace.legacyCommands.length === 0],
    ["mockFallbackZero", trace.mockFallbackHitCount === 0],
    ["screenshotWritten", facts.screenshotWritten === true],
  ];
  const failed = checks.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(
      `SETTINGS Environment evidence failed: ${failed.join(", ")}`,
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

export function applyFailedSettingsEnvironmentEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-environment-current-read-fixture";
  summary.nextAction =
    "Fix the Environment current Host read path and rerun with the same candidate run-id.";
  summary.error = String(error instanceof Error ? error.message : error).slice(
    0,
    500,
  );
  return summary;
}
