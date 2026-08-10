import path from "node:path";

export const MEMORY_SOUL_SCENARIO_ID = "memory-soul-persistence";
export const MEMORY_SOUL_PROFILE_ID = "cheeky_sassy_executor";
export const MEMORY_SOUL_REQUIRED_METHODS = ["soulStylePack/list"];
export const MEMORY_SOUL_REQUIRED_CONFIG_METHODS = [
  "config/read",
  "config/batchWrite",
];
export const MEMORY_SOUL_RUNTIME_MARKERS = [
  "hasInteractionSoul",
  "hasMemorySoulSchema",
  "hasSavedConfigSource",
  "hasProfileId",
  "hasStylePack",
  "hasResponseContract",
  "hasToolLifecycleSurfaceContracts",
  "hasAllowedStyleMoves",
  "hasForbiddenStyleMoves",
];

const APP_SERVER_COMMAND = "app_server_handle_json_lines";
const LEGACY_MEMORY_COMMANDS = [
  "get_memory_config",
  "save_memory_config",
  "memory_soul_read",
  "memory_soul_write",
  "get_soul_config",
  "save_soul_config",
];
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

function validateName(value, label) {
  const normalized = String(value ?? "").trim();
  if (!SAFE_NAME.test(normalized)) {
    throw new Error(`invalid Settings Memory Soul ${label}`);
  }
  return normalized;
}

export function parseSettingsMemorySoulFixtureArgs(
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
    if (arg === "--profile-id" && next) {
      options.profileId = next.trim();
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
      `standalone-settings-memory-soul-${new Date()
        .toISOString()
        .replace(/[-:.]/g, "")}-${process.pid}`,
    "run-id",
  );
  options.prefix = validateName(options.prefix, "prefix");
  options.profileId = validateName(
    options.profileId || MEMORY_SOUL_PROFILE_ID,
    "profile-id",
  );
  if (!options.evidenceDir) {
    options.evidenceDir = path.join(
      cwd,
      ".lime",
      "qc",
      "project-gates",
      options.runId,
      "settings-memory-soul-persistence",
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

function parseRequestMethods(entries) {
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

export function summarizeSettingsMemorySoulTrace(traceRaws) {
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
  const methods = parseRequestMethods(appServerIpcEntries);
  return {
    appServerIpcHitCount: appServerIpcEntries.length,
    methods,
    missingMethods: MEMORY_SOUL_REQUIRED_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    configMethods: methods,
    missingConfigMethods: MEMORY_SOUL_REQUIRED_CONFIG_METHODS.filter(
      (method) => !methods.includes(method),
    ),
    legacyCommands: LEGACY_MEMORY_COMMANDS.filter((command) =>
      commands.has(command),
    ),
    mockFallbackHitCount: appServerEntries.filter(
      (entry) => entry.transport !== "electron-ipc",
    ).length,
  };
}

function runtimeMarkersComplete(runtime) {
  return (
    runtime?.ok === true &&
    runtime?.scenario === "soul-style" &&
    runtime?.soulStylePromptContextCoveredByRuntime === true &&
    runtime?.soulStyleConfig?.enabled === true &&
    runtime?.soulStyleConfig?.style_profile_id ===
      runtime?.soulStyleExpectation?.profileId &&
    MEMORY_SOUL_RUNTIME_MARKERS.every(
      (key) => runtime?.soulStylePromptContextMarkers?.[key] === true,
    )
  );
}

export function createSettingsMemorySoulEvidence({
  candidateRunId,
  startedAt,
  prefix,
  profileId,
}) {
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-memory-soul-persistence",
    priority: "P0",
    proofLevel: "Gate B-R",
    claimBoundary:
      "Real Electron Memory/Soul Settings save and cold-restart recovery through App Server config/read and config/batchWrite plus current soulStylePack/list, combined with a separate isolated current soul-style runtime fixture using the same canonical profile to prove RuntimeCore prompt marker injection. It does not store Soul text, prompt text, user content, paths, provider requests, or secrets, and it does not claim the GUI and runtime launches share one process or app-data directory.",
    candidateRunId: validateName(candidateRunId, "run-id"),
    testOnly: true,
    startedAt,
    result: "fail",
    failureClass: "settings-memory-soul-not-completed",
    nextAction: "Run the real Electron Memory/Soul persistence fixture.",
    settingsScenarioProof: {
      scenarioId: "memory-soul-persistence",
      complete: false,
    },
    profileId: validateName(profileId, "profile-id"),
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
      savedScreenshot: `${prefix}-saved.png`,
      recoveredScreenshot: `${prefix}-recovered.png`,
      finalScreenshot: `${prefix}-recovered.png`,
      screenshot: `${prefix}-recovered.png`,
      rawEvidence: `${prefix}-raw.json`,
      summary: `${prefix}-summary.json`,
    },
  };
}

export function applyPassingSettingsMemorySoulEvidence(summary, facts) {
  const { trace, consoleErrors, pageErrors } = facts;
  summary.bridge = {
    electron: facts.electronLaunchCount === 2,
    preloadInvoke: facts.preloadLaunchCount === 2,
    transport: trace.appServerIpcHitCount > 0 ? "electron-ipc" : null,
    command: APP_SERVER_COMMAND,
    appServerIpcHitCount: trace.appServerIpcHitCount,
    methods: trace.methods,
  };
  summary.config = {
    methods: trace.configMethods,
    requiredMethodsComplete: trace.missingConfigMethods.length === 0,
  };
  summary.lifecycle = {
    isolatedUserData: facts.isolatedUserData === true,
    guiSaved: facts.guiSaved === true,
    restartReadback: facts.restartReadback === true,
    memoryEnabled: facts.memoryEnabled === true,
    soulEnabled: facts.soulEnabled === true,
    profileSelected: facts.profileSelected === true,
    runtimePromptMarkers: runtimeMarkersComplete(facts.runtime),
    runtimeProfileId: facts.runtime?.soulStyleExpectation?.profileId ?? null,
  };
  summary.runtime = {
    proofLevel: facts.runtime?.proofLevel ?? null,
    profileId: facts.runtime?.soulStyleExpectation?.profileId ?? null,
    markerKeys: MEMORY_SOUL_RUNTIME_MARKERS,
    markersComplete: runtimeMarkersComplete(facts.runtime),
    promptStored: false,
    providerRequestStored: false,
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
    ["twoRealElectronLaunches", summary.bridge.electron],
    ["twoPreloadInvokeBridges", summary.bridge.preloadInvoke],
    ["appServerElectronIpc", summary.bridge.appServerIpcHitCount > 0],
    ["currentSoulStyleMethod", trace.missingMethods.length === 0],
    ["configMethods", summary.config.requiredMethodsComplete],
    ["isolatedUserData", summary.lifecycle.isolatedUserData],
    ["guiSaved", summary.lifecycle.guiSaved],
    ["restartReadback", summary.lifecycle.restartReadback],
    ["memoryEnabled", summary.lifecycle.memoryEnabled],
    ["soulEnabled", summary.lifecycle.soulEnabled],
    ["profileSelected", summary.lifecycle.profileSelected],
    ["runtimePromptMarkers", summary.lifecycle.runtimePromptMarkers],
    ["runtimeProfileMatches", summary.runtime.profileId === summary.profileId],
    ["consoleErrorsZero", consoleErrors.length === 0],
    ["pageErrorsZero", pageErrors.length === 0],
    ["invokeErrorsZero", facts.invokeErrorCount === 0],
    ["legacyCommandsZero", trace.legacyCommands.length === 0],
    ["mockFallbackZero", trace.mockFallbackHitCount === 0],
    ["savedScreenshotWritten", facts.savedScreenshotWritten === true],
    ["recoveredScreenshotWritten", facts.recoveredScreenshotWritten === true],
  ];
  const failed = checks.filter(([, passed]) => !passed).map(([name]) => name);
  if (failed.length > 0) {
    throw new Error(
      `SETTINGS Memory Soul evidence failed: ${failed.join(", ")}`,
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

export function applyFailedSettingsMemorySoulEvidence(summary, error) {
  summary.result = "fail";
  summary.settingsScenarioProof.complete = false;
  summary.assertions = {
    total: 1,
    passed: 0,
    failed: ["scenarioFailed"],
    details: {},
  };
  summary.failureClass = "settings-memory-soul-fixture";
  summary.nextAction =
    "Fix the Memory/Soul current lifecycle and rerun with the same candidate run-id using a new prefix.";
  summary.errorClass =
    error instanceof Error && error.name ? error.name : "Error";
  return summary;
}
