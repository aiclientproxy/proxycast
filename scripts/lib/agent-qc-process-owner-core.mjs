const GUI_OWNER_PATTERNS = [
  "verify:gui-smoke",
  "smoke:electron",
  "scripts/electron/smoke.mjs",
  "electron-smoke.mjs",
  "smoke:workspace-ready",
  "smoke:site-adapters",
  "smoke:agent-service-skill-entry",
  "smoke:agent-runtime-tool-surface",
  "smoke:knowledge-gui",
  "smoke:design-canvas",
  "claw-chat-ready-streaming",
  "workspace-ready-session-restore",
  "release-package-startup-smoke",
];

const CARGO_OWNER_PATTERNS = ["cargo ", "cargo-fmt", "rustc ", "clippy-driver"];

const QCLOOP_OWNER_PATTERNS = [
  "qcloop --db",
  "./qcloop --db",
  "qcloop serve",
  "qcloop-worker",
  "qcloop_worker_result",
  "agent qc p0",
  "只读执行 lime agent qc p0",
];

const OBSERVER_PROCESS_PATTERNS = [
  "while ps -p",
  "while ps aux | rg",
  "ps -eo pid,ppid,etime,pcpu,pmem,command | rg",
  "ps aux | rg",
];

const PASSIVE_QCLOOP_SERVER_PATTERNS = [
  "qcloop --db",
  "./qcloop --db",
  "qcloop serve",
];

const ACTIVE_QCLOOP_WORKER_PATTERNS = [
  "qcloop-worker",
  "qcloop_worker_result",
  "agent qc p0",
  "只读执行 lime agent qc p0",
];

const PASSIVE_ELECTRON_RUNTIME_PATTERNS = [
  "electron:dev",
  "run-electron-dev.mjs",
  "scripts/electron/run-dev.mjs",
  "node_modules/.bin/electron .",
  "node_modules/electron/dist/electron .",
  "node_modules/electron/dist/electron.app/",
  ".lime/electron-dev-host/lime.app/",
  ".lime/electron-dev-host/lime-dev.app/",
];

function parseEtimeSeconds(value) {
  const text = String(value || "").trim();
  if (!text) {
    return null;
  }
  const dayParts = text.split("-");
  const dayCount = dayParts.length === 2 ? Number(dayParts[0]) : 0;
  const timeText = dayParts.length === 2 ? dayParts[1] : dayParts[0];
  const parts = timeText.split(":").map((part) => Number(part));
  if (parts.some((part) => !Number.isFinite(part))) {
    return null;
  }
  if (parts.length === 3) {
    return dayCount * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  if (parts.length === 2) {
    return dayCount * 86400 + parts[0] * 60 + parts[1];
  }
  if (parts.length === 1) {
    return dayCount * 86400 + parts[0];
  }
  return null;
}

function sanitizeProcessCommand(command) {
  return String(command || "")
    .replace(/(--api-key(?:=|\s+))(?:"[^"]+"|'[^']+'|\S+)/gi, "$1<redacted>")
    .replace(
      /(api[_-]?key(?:=|:|\s+))(?:"[^"]+"|'[^']+'|\S+)/gi,
      "$1<redacted>",
    )
    .replace(/ctx7sk-[A-Za-z0-9-]+/g, "ctx7sk-***")
    .replace(/sk-[A-Za-z0-9_-]{12,}/g, "sk-***");
}

function commandHasAny(command, patterns) {
  const normalized = String(command || "").toLowerCase();
  return patterns.some((pattern) => normalized.includes(pattern.toLowerCase()));
}

function isObserverProcess(entry) {
  return commandHasAny(entry?.command, OBSERVER_PROCESS_PATTERNS);
}

function isPassiveQcloopServer(entry) {
  return (
    commandHasAny(entry?.command, PASSIVE_QCLOOP_SERVER_PATTERNS) &&
    !commandHasAny(entry?.command, ACTIVE_QCLOOP_WORKER_PATTERNS)
  );
}

function isPassiveElectronRuntime(entry) {
  return commandHasAny(entry?.command, PASSIVE_ELECTRON_RUNTIME_PATTERNS);
}

function isPassiveDesktopRuntime(entry) {
  return isPassiveElectronRuntime(entry);
}

function uniqueByPid(entries) {
  const seen = new Set();
  const result = [];
  for (const entry of entries) {
    if (seen.has(entry.pid)) {
      continue;
    }
    seen.add(entry.pid);
    result.push(entry);
  }
  return result.sort((left, right) => left.pid - right.pid);
}

function normalizeProcessEntry(entry) {
  const etime = String(entry?.etime || "unknown");
  return {
    ...entry,
    pid: Number(entry?.pid || 0),
    ppid: Number(entry?.ppid || 0),
    pgid: entry?.pgid ?? null,
    stat: entry?.stat || "unknown",
    etime,
    durationSeconds: entry?.durationSeconds ?? parseEtimeSeconds(etime),
    command: sanitizeProcessCommand(entry?.command || ""),
  };
}

function createAgentQcProcessOwnerReport(
  processes,
  {
    generatedAt = new Date().toISOString(),
    platform = process.platform,
    maxActiveGuiSmoke = 0,
    maxCargoOrRust = 0,
    maxQcloopRelated = 0,
    staleMinutes = 30,
  } = {},
) {
  const normalizedProcesses = processes
    .map(normalizeProcessEntry)
    .filter((entry) => entry.pid > 0 && entry.command.length > 0);
  const observerProcesses = uniqueByPid(
    normalizedProcesses.filter(isObserverProcess),
  );
  const ownerCandidateProcesses = normalizedProcesses.filter(
    (entry) => !isObserverProcess(entry),
  );
  const passiveQcloopServerProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(isPassiveQcloopServer),
  );
  const passiveElectronRuntimeProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(isPassiveElectronRuntime),
  );
  const passiveDesktopRuntimeProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(isPassiveDesktopRuntime),
  );
  const activeGuiSmokeProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(
      (entry) =>
        commandHasAny(entry.command, GUI_OWNER_PATTERNS) &&
        !isPassiveDesktopRuntime(entry),
    ),
  );
  const qcloopProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(
      (entry) =>
        commandHasAny(entry.command, QCLOOP_OWNER_PATTERNS) &&
        !isPassiveQcloopServer(entry),
    ),
  );
  const cargoProcesses = uniqueByPid(
    ownerCandidateProcesses.filter(
      (entry) =>
        commandHasAny(entry.command, CARGO_OWNER_PATTERNS) &&
        !isPassiveDesktopRuntime(entry),
    ),
  );
  const staleThresholdSeconds = Math.max(0, Number(staleMinutes) || 0) * 60;
  const staleActiveGuiSmokeProcesses = uniqueByPid(
    activeGuiSmokeProcesses.filter(
      (entry) => Number(entry.durationSeconds || 0) >= staleThresholdSeconds,
    ),
  );

  const counts = {
    activeGuiSmoke: activeGuiSmokeProcesses.length,
    cargoOrRust: cargoProcesses.length,
    qcloopRelated: qcloopProcesses.length,
    staleActiveGuiSmoke: staleActiveGuiSmokeProcesses.length,
    passiveQcloopServer: passiveQcloopServerProcesses.length,
    passiveElectronRuntime: passiveElectronRuntimeProcesses.length,
    passiveDesktopRuntime: passiveDesktopRuntimeProcesses.length,
    observer: observerProcesses.length,
  };
  const busy =
    counts.activeGuiSmoke > maxActiveGuiSmoke ||
    counts.cargoOrRust > maxCargoOrRust ||
    counts.qcloopRelated > maxQcloopRelated;

  return {
    schemaVersion: "v1",
    generatedAt,
    platform,
    maxActiveGuiSmoke,
    maxCargoOrRust,
    maxQcloopRelated,
    staleMinutes,
    verdict: {
      status: busy ? "busy" : "pass",
      summary: `activeGuiSmoke=${counts.activeGuiSmoke}, cargoOrRust=${counts.cargoOrRust}, qcloopRelated=${counts.qcloopRelated}, staleActiveGuiSmoke=${counts.staleActiveGuiSmoke}, passiveQcloopServer=${counts.passiveQcloopServer}, passiveElectronRuntime=${counts.passiveElectronRuntime}, passiveDesktopRuntime=${counts.passiveDesktopRuntime}, observer=${counts.observer}`,
      nextAction: busy
        ? "Do not start full verify:local or another GUI P0 batch while these owners are active; continue read-only observation or wait for natural completion."
        : "No active raw GUI smoke, Cargo/Rust, or qcloop owner was observed; heavy gates may run if qcloop GUI owner and release evidence gates are also clear.",
    },
    ownerIntervention: staleActiveGuiSmokeProcesses.length
      ? {
          status: "requires_owner_confirmation",
          processIds: staleActiveGuiSmokeProcesses.map((entry) => entry.pid),
          requiredConfirmationText: `确认处理 stale raw GUI owner PID ${staleActiveGuiSmokeProcesses.map((entry) => entry.pid).join(", ")}，可以终止这些进程并记录 sidecar。`,
          prohibitedUntilConfirmed: [
            "kill / pause / interrupt stale raw GUI owner",
            "start full verify:local",
            "start another full GUI P0 batch",
            "overwrite .lime/qc/agent-qc-evidence.json",
            "git commit / push / tag / release",
          ],
          nextAction: "等待这些进程自然释放，或由 owner 明确确认后再处理。",
        }
      : {
          status: "not_required",
          processIds: [],
          requiredConfirmationText: "",
          prohibitedUntilConfirmed: [],
          nextAction: busy
            ? "等待 active owner 自然释放。"
            : "无需 stale raw owner 处置。",
        },
    activeGuiSmokeProcesses,
    qcloopProcesses,
    cargoProcesses,
    staleActiveGuiSmokeProcesses,
    passiveQcloopServerProcesses,
    passiveElectronRuntimeProcesses,
    passiveDesktopRuntimeProcesses,
    observerProcesses,
    guardrails: [
      "best-effort process snapshot only",
      "do not kill or restart listed processes from this sidecar",
      "use this sidecar to decide whether heavy GUI or verify gates should wait",
    ],
  };
}

export {
  createAgentQcProcessOwnerReport,
  parseEtimeSeconds,
  sanitizeProcessCommand,
};
