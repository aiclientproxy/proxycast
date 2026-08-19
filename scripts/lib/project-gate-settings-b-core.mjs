const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export const SETTINGS_GATE_B_SCENARIOS = Object.freeze([
  { id: "home-navigation", tab: "home", owner: "settings" },
  { id: "profile-persistence", tab: "profile", owner: "settings" },
  { id: "stats-current-read", tab: "stats", owner: "settings" },
  { id: "appearance-persistence", tab: "appearance", owner: "settings" },
  { id: "memory-ready", tab: "memory", owner: "settings/app-server" },
  {
    id: "memory-soul-persistence",
    tab: "memory",
    owner: "settings/app-server",
  },
  {
    id: "archived-lifecycle-recovery",
    tab: "archived-conversations",
    owner: "settings/app-server",
  },
  {
    id: "provider-crud-model-auth",
    tab: "providers",
    owner: "settings/model-provider",
  },
  {
    id: "media-services-readiness",
    tab: "media-services",
    owner: "settings/model-provider",
  },
  { id: "mcp-create-list", tab: "mcp-server", owner: "settings/mcp" },
  {
    id: "mcp-lifecycle-recovery",
    tab: "mcp-server",
    owner: "settings/mcp",
  },
  { id: "web-search-route", tab: "web-search", owner: "settings" },
  {
    id: "environment-current-read",
    tab: "environment",
    owner: "settings/desktop-host",
  },
  {
    id: "execution-policy-allow-deny-error",
    tab: "execution-policy",
    owner: "settings/app-server",
  },
  {
    id: "automation-lifecycle",
    tab: "automation",
    owner: "settings/app-server",
  },
  {
    id: "developer-current-diagnostics",
    tab: "developer",
    owner: "settings/desktop-host/app-server",
  },
  {
    id: "about-version-truth",
    tab: "about",
    owner: "settings/desktop-host",
  },
]);

const SCENARIO_IDS = new Set(
  SETTINGS_GATE_B_SCENARIOS.map((scenario) => scenario.id),
);

export function validateSettingsGateBRunId(value) {
  const runId = String(value ?? "").trim();
  if (!RUN_ID_PATTERN.test(runId)) {
    throw new Error("invalid SETTINGS-01 Gate B run-id");
  }
  return runId;
}

function assertionsPass(assertions) {
  return Boolean(
    assertions &&
    Number.isInteger(assertions.total) &&
    assertions.total > 0 &&
    assertions.passed === assertions.total &&
    Array.isArray(assertions.failed) &&
    assertions.failed.length === 0,
  );
}

function arrayEmpty(value) {
  return Array.isArray(value) && value.length === 0;
}

function assertCommonSource(summary, candidateRunId, label) {
  if (summary?.schemaVersion !== 1) {
    throw new Error(`${label}: schemaVersion must be 1`);
  }
  if (summary.candidateRunId !== candidateRunId) {
    throw new Error(`${label}: candidateRunId mismatch`);
  }
  if (summary.result !== "pass" || !assertionsPass(summary.assertions)) {
    throw new Error(`${label}: source evidence did not pass`);
  }
}

function adaptShellMemory(summary, candidateRunId) {
  const label = "shell-memory";
  assertCommonSource(summary, candidateRunId, label);
  const details = summary.assertions.details ?? {};
  const bridge = summary.bridge ?? {};
  const errors = summary.errors ?? {};
  const requiredZeroErrors = [
    "consoleErrorCount",
    "pageErrorCount",
    "invokeErrorCount",
    "traceErrorCount",
    "rendererCrashCount",
    "rendererUnresponsiveCount",
    "preloadErrorCount",
    "rendererLoadErrorCount",
    "legacyCommandHitCount",
    "mockFallbackHitCount",
  ];
  const settingsRoute = summary.routes?.find(
    (route) => route?.stage === "settings-memory",
  );
  const valid =
    summary.surfaceProof?.surfaceId === "SHELL-01" &&
    summary.surfaceProof?.proof === "gate-b-f" &&
    summary.surfaceProof?.complete === true &&
    details.settingsMemoryReady === true &&
    settingsRoute?.ready === true &&
    bridge.electron === true &&
    bridge.preloadInvoke === true &&
    bridge.transport === "electron-ipc" &&
    bridge.command === "app_server_handle_json_lines" &&
    bridge.appServerIpcHitCount > 0 &&
    Array.isArray(bridge.methods) &&
    bridge.methods.length > 0 &&
    bridge.hostInitialized === true &&
    requiredZeroErrors.every((key) => errors[key] === 0) &&
    arrayEmpty(errors.legacyCommands) &&
    typeof summary.artifacts?.screenshot === "string" &&
    typeof summary.artifacts?.trace === "string";
  if (!valid) {
    throw new Error(`${label}: incomplete real Electron Memory evidence`);
  }
  return {
    scenarioId: "memory-ready",
    sourceKind: label,
    methods: [...bridge.methods],
  };
}

function adaptSettingsScenario(summary, candidateRunId) {
  const label = "settings-scenario";
  assertCommonSource(summary, candidateRunId, label);
  const proof = summary.settingsScenarioProof;
  const bridge = summary.bridge ?? {};
  const errors = summary.errors ?? {};
  const scenarioId = proof?.scenarioId;
  const valid =
    SCENARIO_IDS.has(scenarioId) &&
    proof?.complete === true &&
    ["Gate B-F", "Gate B-R"].includes(summary.proofLevel) &&
    bridge.electron === true &&
    bridge.preloadInvoke === true &&
    bridge.transport === "electron-ipc" &&
    bridge.command === "app_server_handle_json_lines" &&
    bridge.appServerIpcHitCount > 0 &&
    Array.isArray(bridge.methods) &&
    bridge.methods.length > 0 &&
    errors.consoleErrorCount === 0 &&
    errors.pageErrorCount === 0 &&
    errors.invokeErrorCount === 0 &&
    errors.legacyCommandHitCount === 0 &&
    arrayEmpty(errors.legacyCommands) &&
    errors.mockFallbackHitCount === 0 &&
    typeof summary.artifacts?.screenshot === "string";
  if (!valid) {
    throw new Error(`${label}: incomplete ${String(scenarioId)} evidence`);
  }
  return {
    scenarioId,
    sourceKind: label,
    methods: [...bridge.methods],
  };
}

export function adaptSettingsGateBSource(record, candidateRunId) {
  const runId = validateSettingsGateBRunId(candidateRunId);
  if (!record || typeof record !== "object") {
    throw new Error("SETTINGS-01 Gate B source record must be an object");
  }
  if (record.kind === "shell-memory") {
    return adaptShellMemory(record.value, runId);
  }
  if (record.kind === "settings-scenario") {
    return adaptSettingsScenario(record.value, runId);
  }
  throw new Error(`unsupported SETTINGS-01 Gate B source kind: ${record.kind}`);
}

export function buildSettingsGateBFEvidence({
  candidateRunId,
  startedAt,
  completedAt,
  sourceRecords,
}) {
  const runId = validateSettingsGateBRunId(candidateRunId);
  if (!Array.isArray(sourceRecords) || sourceRecords.length === 0) {
    throw new Error("SETTINGS-01 Gate B requires at least one source record");
  }
  const adapted = sourceRecords.map((record) => ({
    ...adaptSettingsGateBSource(record, runId),
    file: record.file,
    sha256: record.sha256,
  }));
  for (const source of adapted) {
    if (
      typeof source.file !== "string" ||
      !source.file ||
      pathLooksUnsafe(source.file) ||
      !/^[a-f0-9]{64}$/.test(source.sha256 ?? "")
    ) {
      throw new Error(
        `invalid SETTINGS-01 Gate B source artifact: ${source.file}`,
      );
    }
  }
  const duplicates = adapted
    .map((entry) => entry.scenarioId)
    .filter(
      (scenarioId, index, values) => values.indexOf(scenarioId) !== index,
    );
  if (duplicates.length > 0) {
    throw new Error(
      `duplicate SETTINGS-01 Gate B scenarios: ${Array.from(new Set(duplicates)).join(", ")}`,
    );
  }
  const completedSet = new Set(adapted.map((entry) => entry.scenarioId));
  const completedScenarios = SETTINGS_GATE_B_SCENARIOS.filter((scenario) =>
    completedSet.has(scenario.id),
  ).map((scenario) => scenario.id);
  const missingScenarios = SETTINGS_GATE_B_SCENARIOS.filter(
    (scenario) => !completedSet.has(scenario.id),
  ).map((scenario) => scenario.id);
  const complete = missingScenarios.length === 0;

  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-gate-b-f",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary:
      "Aggregation of same-run real Electron owner evidence for all current Settings tabs. It does not claim live-provider, packaged-app, or unexecuted scenarios.",
    candidateRunId: runId,
    testOnly: true,
    startedAt,
    completedAt,
    result: "pass",
    failureClass: null,
    nextAction: complete
      ? null
      : `Complete the missing SETTINGS-01 Gate B-F scenarios: ${missingScenarios.join(", ")}`,
    surfaceProof: {
      surfaceId: "SETTINGS-01",
      proof: "gate-b-f",
      complete,
    },
    missingScenarios,
    coverage: {
      requiredScenarios: SETTINGS_GATE_B_SCENARIOS,
      completedScenarios,
      completed: completedScenarios.length,
      total: SETTINGS_GATE_B_SCENARIOS.length,
      scenarioStatus: Object.fromEntries(
        SETTINGS_GATE_B_SCENARIOS.map((scenario) => [
          scenario.id,
          completedSet.has(scenario.id),
        ]),
      ),
    },
    assertions: {
      total: completedScenarios.length,
      passed: completedScenarios.length,
      failed: [],
      details: Object.fromEntries(
        completedScenarios.map((scenarioId) => [scenarioId, true]),
      ),
    },
    sources: adapted.map((entry) => ({
      scenarioId: entry.scenarioId,
      sourceKind: entry.sourceKind,
      file: entry.file,
      sha256: entry.sha256,
      methods: entry.methods,
    })),
    artifacts: { summary: "summary.json" },
  };
}

function pathLooksUnsafe(value) {
  return (
    value.startsWith("/") ||
    value.split(/[\\/]+/u).some((segment) => segment === "..")
  );
}

export function buildSettingsGateBFailureEvidence({
  candidateRunId,
  startedAt,
  completedAt,
  error,
}) {
  const runId = validateSettingsGateBRunId(candidateRunId);
  return {
    schemaVersion: 1,
    scenarioId: "SETTINGS-01-gate-b-f",
    priority: "P0",
    proofLevel: "Gate B-F",
    claimBoundary: "SETTINGS-01 Gate B-F aggregation failed before completion.",
    candidateRunId: runId,
    testOnly: true,
    startedAt,
    completedAt,
    result: "fail",
    failureClass: "settings-gate-b-evidence-aggregation",
    nextAction:
      "Fix the invalid or failed owner evidence and rerun the SETTINGS-01 Gate B-F aggregator with the same candidate run-id.",
    surfaceProof: {
      surfaceId: "SETTINGS-01",
      proof: "gate-b-f",
      complete: false,
    },
    missingScenarios: SETTINGS_GATE_B_SCENARIOS.map((scenario) => scenario.id),
    assertions: { total: 1, passed: 0, failed: ["aggregationFailed"] },
    error: String(error instanceof Error ? error.message : error).slice(0, 500),
    artifacts: { summary: "summary.json" },
  };
}
