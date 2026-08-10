import path from "node:path";

export const DEVELOPER_SCENARIO_ID = "developer-current-diagnostics";
export const DEVELOPER_REQUIRED_APP_SERVER_METHODS = [
  "log/list",
  "log/persistedTail",
  "diagnostics/server/read",
  "diagnostics/logStorage/read",
  "diagnostics/windowsStartup/read",
  "modelProvider/list",
  "mcpServerStatus/list",
];
export const DEVELOPER_REQUIRED_CONFIG_METHODS = ["config/read"];
export const DEVELOPER_REQUIRED_HOST_COMMANDS = [];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const LEGACY_DEVELOPER_COMMANDS = [
  "get_logs",
  "get_persisted_logs_tail",
  "get_server_diagnostics",
  "get_log_storage_diagnostics",
  "get_windows_startup_diagnostics",
  "get_api_key_providers",
  "mcp_list_servers_with_status",
];
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings Developer ${label}`);
  }
  return normalized;
}

export function parseSettingsDeveloperFixtureArgs(
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
      `standalone-settings-developer-${new Date()
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
      "settings-developer-current-diagnostics",
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

export function summarizeSettingsDeveloperTrace(traceRaw) {
  const entries = parseTrace(traceRaw);
  const appServerEntries = entries.filter(
    (entry) => entry?.command === APP_SERVER_COMMAND,
  );
  const appServerIpcEntries = appServerEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const methods = appServerMethods(appServerIpcEntries);
  const hostEntries = [];
  const hostIpcEntries = hostEntries.filter(
    (entry) => entry.transport === "electron-ipc",
  );
  const hostCommands = Array.from(
    new Set(hostIpcEntries.map((entry) => entry.command)),
  );
  const commands = new Set(entries.map((entry) => entry?.command));
  return {
    appServerIpcHitCount: appServerIpcEntries.length,
    methods,
    missingMethods: DEVELOPER_REQUIRED_APP_SERVER_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    hostIpcHitCount: hostIpcEntries.length,
    hostCommands,
    missingConfigMethods: DEVELOPER_REQUIRED_CONFIG_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    legacyCommands: LEGACY_DEVELOPER_COMMANDS.filter((command) =>
      commands.has(command),
    ),
    mockFallbackHitCount: [...appServerEntries, ...hostEntries].filter(
      (entry) => entry.transport !== "electron-ipc",
    ).length,
  };
}

export function createSettingsDeveloperEvidence({
  candidateRunId,
  startedAt,
  prefix,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-developer-current-diagnostics",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary:
      "Real Electron Developer diagnostic collection through current Host and App Server methods. A renderer test-only clipboard sink replaces only the final operating-system clipboard write and records JSON shape booleans plus text length; it does not replace collection. Evidence stores no clipboard content, log text, config, path, provider data, MCP data, or secret, and does not claim operating-system clipboard delivery.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-developer-not-completed",
    nextAction:
      "Run the real Electron Developer diagnostic collection fixture to a terminal copied state.",
    settingsScenarioProof: {
      scenarioId: DEVELOPER_SCENARIO_ID,
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

export function applyPassingSettingsDeveloperEvidence(summary, facts) {
  const { trace, consoleErrors, pageErrors, clipboard } = facts;
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
  summary.clipboardSink = {
    testOnly: true,
    writeCount: clipboard.writeCount,
    textLength: clipboard.textLength,
    jsonObject: clipboard.jsonObject,
    payloadShape: clipboard.payloadShape,
  };
  summary.errors = {
    consoleErrorCount: consoleErrors.length,
    pageErrorCount: pageErrors.length,
    invokeErrorCount: facts.invokeErrorCount,
    legacyCommandHitCount: trace.legacyCommands.length,
    legacyCommands: trace.legacyCommands,
    mockFallbackHitCount: trace.mockFallbackHitCount,
  };
  const payloadShapeReady = Object.values(clipboard.payloadShape ?? {}).every(
    (value) => value === true,
  );
  const checks = [
    ["realElectronRenderer", summary.bridge.electron],
    ["preloadInvokeBridge", summary.bridge.preloadInvoke],
    ["appServerElectronIpc", summary.bridge.appServerIpcHitCount > 0],
    ["allCurrentDiagnosticMethods", trace.missingMethods.length === 0],
    ["configCurrentRead", trace.missingConfigMethods.length === 0],
    ["isolatedUserData", facts.isolatedUserData === true],
    ["developerTabActive", facts.developerTabActive === true],
    ["developerLabActive", facts.developerLabActive === true],
    ["copyJsonActionReady", facts.copyJsonActionReady === true],
    ["diagnosticTerminalSuccess", facts.diagnosticSuccess === true],
    ["clipboardSinkInstalled", facts.clipboardSinkInstalled === true],
    ["singleClipboardWrite", clipboard.writeCount === 1],
    ["clipboardTextNonEmpty", clipboard.textLength > 0],
    ["clipboardJsonObject", clipboard.jsonObject === true],
    ["diagnosticPayloadShape", payloadShapeReady],
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
    throw new Error(`SETTINGS Developer evidence failed: ${failed.join(", ")}`);
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

export function applyFailedSettingsDeveloperEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-developer-current-diagnostics-fixture";
  summary.nextAction =
    "Fix the Developer current diagnostic collection path and rerun with the same candidate run-id.";
  summary.error = String(error instanceof Error ? error.message : error).slice(
    0,
    500,
  );
  return summary;
}
