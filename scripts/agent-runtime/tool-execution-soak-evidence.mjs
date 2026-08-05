import { execFileSync } from "node:child_process";
import path from "node:path";
import process from "node:process";

const TERMINAL_TURN_STATUSES = new Set(["canceled", "completed", "failed"]);
const RSS_GROWTH_BUDGET_KB = 512 * 1024;
const ROUND_DURATION_BUDGET_MS = 120_000;

function integerArg(args, name, fallback) {
  const index = args.indexOf(name);
  if (index < 0) return fallback;
  const value = Number(args[index + 1]);
  if (!Number.isInteger(value)) {
    throw new Error(`${name} 必须是整数`);
  }
  return value;
}

export function resolveSoakConfig(args) {
  const rounds = integerArg(args, "--soak-rounds", 1);
  const coldRestart = args.includes("--cold-restart");
  const coldRestarts = integerArg(args, "--cold-restarts", coldRestart ? 1 : 0);
  if (rounds < 1 || rounds > 20) {
    throw new Error("--soak-rounds 必须是 1 到 20 的整数");
  }
  if (coldRestarts < 0 || coldRestarts > 10) {
    throw new Error("--cold-restarts 必须是 0 到 10 的整数");
  }
  if (!coldRestart && coldRestarts > 0) {
    throw new Error("--cold-restarts 需要同时启用 --cold-restart");
  }
  if (rounds > 1 && coldRestarts < 2) {
    throw new Error("SOAK 多轮模式至少需要两次 cold restart");
  }
  return {
    enabled: rounds > 1,
    rounds,
    coldRestarts,
  };
}

export function roundEvidencePath(outputPath, roundIndex, roundCount) {
  if (roundIndex === roundCount - 1) return outputPath;
  const extension = path.extname(outputPath);
  const stem = extension ? outputPath.slice(0, -extension.length) : outputPath;
  return `${stem}-round-${String(roundIndex + 1).padStart(2, "0")}${extension || ".json"}`;
}

export function childArgsForRound(args, outputPath) {
  const childArgs = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--cold-restart") continue;
    if (["--cold-restarts", "--soak-rounds", "--output"].includes(arg)) {
      index += 1;
      continue;
    }
    childArgs.push(arg);
  }
  return [...childArgs, "--output", outputPath];
}

export function parsePosixProcessRows(output) {
  return String(output || "")
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
      if (!match) return null;
      return {
        pid: Number(match[1]),
        ppid: Number(match[2]),
        rssKb: Number(match[3]),
        command: match[4],
      };
    })
    .filter(Boolean);
}

export function parseWindowsProcessRows(output) {
  const parsed = JSON.parse(String(output || "[]"));
  return (Array.isArray(parsed) ? parsed : [parsed]).map((row) => ({
    pid: Number(row.ProcessId),
    ppid: Number(row.ParentProcessId),
    rssKb: Math.round(Number(row.WorkingSetSize || 0) / 1024),
    command: String(row.CommandLine || row.Name || ""),
  }));
}

function readSystemProcessRows({
  platform = process.platform,
  execFile = execFileSync,
} = {}) {
  if (platform === "win32") {
    const output = execFile(
      "powershell.exe",
      [
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,WorkingSetSize,Name,CommandLine | ConvertTo-Json -Compress",
      ],
      { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
    );
    return parseWindowsProcessRows(output);
  }
  const output = execFile("ps", ["-axo", "pid=,ppid=,rss=,command="], {
    encoding: "utf8",
    maxBuffer: 16 * 1024 * 1024,
  });
  return parsePosixProcessRows(output);
}

function processRole(row, rootPid) {
  if (row.pid === rootPid) return "electron-host";
  if (/(^|[\\/\s])app-server(?:\.exe)?(?:\s|$)/i.test(row.command)) {
    return "app-server";
  }
  return "electron-child";
}

export function summarizeProcessTree(rows, rootPid, stage = "unknown") {
  const descendants = new Set([rootPid]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const row of rows) {
      if (!descendants.has(row.pid) && descendants.has(row.ppid)) {
        descendants.add(row.pid);
        changed = true;
      }
    }
  }
  const processes = rows
    .filter((row) => descendants.has(row.pid))
    .map((row) => ({
      pid: row.pid,
      ppid: row.ppid,
      rssKb: row.rssKb,
      role: processRole(row, rootPid),
    }))
    .sort((left, right) => left.pid - right.pid);
  return {
    stage,
    capturedAt: new Date().toISOString(),
    rootPid,
    processCount: processes.length,
    totalRssKb: processes.reduce((total, row) => total + row.rssKb, 0),
    appServerPids: processes
      .filter((row) => row.role === "app-server")
      .map((row) => row.pid),
    appServerRssKb: processes
      .filter((row) => row.role === "app-server")
      .reduce((total, row) => total + row.rssKb, 0),
    processes,
  };
}

export function collectProcessTreeSnapshot(rootPid, stage, options = {}) {
  return summarizeProcessTree(readSystemProcessRows(options), rootPid, stage);
}

export async function waitForProcessIdsExit(
  processIds,
  { intervalMs = 100, timeoutMs = 15_000, ...options } = {},
) {
  const expected = new Set(processIds.filter(Number.isInteger));
  const startedAt = Date.now();
  let remaining = [];
  while (Date.now() - startedAt < timeoutMs) {
    const live = new Set(readSystemProcessRows(options).map((row) => row.pid));
    remaining = [...expected].filter((pid) => live.has(pid));
    if (remaining.length === 0) {
      return {
        exited: true,
        remainingPids: [],
        waitedMs: Date.now() - startedAt,
      };
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return {
    exited: false,
    remainingPids: remaining,
    waitedMs: Date.now() - startedAt,
  };
}

async function invokeAppServer(page, method, params) {
  return await page.evaluate(
    async ({ method, params }) => {
      const invoke = window.electronAPI?.invoke;
      if (typeof invoke !== "function") {
        throw new Error("Electron preload invoke bridge is unavailable");
      }
      const id = `tool-soak-${Date.now()}-${Math.random()}`;
      const response = await invoke("app_server_handle_json_lines", {
        request: {
          lines: [JSON.stringify({ jsonrpc: "2.0", id, method, params })],
        },
      });
      const messages = (Array.isArray(response?.lines) ? response.lines : [])
        .map((line) => {
          try {
            return JSON.parse(line);
          } catch {
            return null;
          }
        })
        .filter(Boolean);
      const reply = messages.find((message) => message?.id === id);
      if (reply?.error) {
        throw new Error(`${method} failed: ${JSON.stringify(reply.error)}`);
      }
      if (!reply || !Object.hasOwn(reply, "result")) {
        throw new Error(`${method} did not return a JSON-RPC result`);
      }
      return reply.result;
    },
    { method, params },
  );
}

function countByStatus(records) {
  const counts = {};
  for (const record of records) {
    const status = String(record?.status || "unknown");
    counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

function countByKind(records) {
  const counts = {};
  for (const record of records) {
    const kind = String(record?.kind || "unknown");
    counts[kind] = (counts[kind] || 0) + 1;
  }
  return counts;
}

export function resolveSoakRoundThreadIdentity(evidence) {
  const sessionId = String(evidence?.runtime?.sessionId || "").trim();
  const threadId = String(evidence?.runtime?.threadId || "").trim();
  if (!sessionId || !threadId) {
    throw new Error(
      "SOAK evidence 缺少 canonical sessionId/threadId，拒绝使用 sessionId 代替 threadId",
    );
  }
  return { sessionId, threadId };
}

export function containsCanonicalThread(threads, { sessionId, threadId }) {
  return threads.some(
    (thread) =>
      thread?.id === threadId && thread?.sessionId === sessionId,
  );
}

export async function collectSoakRoundObservation({
  evidence,
  outputPath,
  page,
  processSnapshot,
  roundIndex,
}) {
  const requestTimings = [];
  const invokeObserved = async (method, params) => {
    const startedAt = Date.now();
    try {
      return await invokeAppServer(page, method, params);
    } finally {
      requestTimings.push({
        method,
        durationMs: Date.now() - startedAt,
      });
    }
  };
  const { sessionId, threadId } = resolveSoakRoundThreadIdentity(evidence);
  const threadRead = await invokeObserved("thread/read", {
    threadId,
    includeTurns: true,
  });
  const canonicalThread = threadRead?.thread;
  if (!canonicalThread || typeof canonicalThread !== "object") {
    throw new Error("thread/read 未返回 canonical thread");
  }
  const [itemsResponse, threadsResponse, turnsResponse] = await Promise.all([
    invokeObserved("thread/items/list", { threadId, limit: 500 }),
    invokeObserved("thread/list", {
      archived: false,
      limit: 500,
      turnsView: "summary",
    }),
    invokeObserved("thread/turns/list", { threadId, limit: 500 }),
  ]);
  const items = Array.isArray(itemsResponse?.data) ? itemsResponse.data : [];
  const threads = Array.isArray(threadsResponse?.data)
    ? threadsResponse.data
    : [];
  const turns = Array.isArray(turnsResponse?.data) ? turnsResponse.data : [];
  const turnIds = turns
    .map((turn) => String(turn?.turnId || ""))
    .filter(Boolean);
  const itemIds = items
    .map((item) => String(item?.itemId || ""))
    .filter(Boolean);
  const terminalTurns = turns.filter((turn) =>
    TERMINAL_TURN_STATUSES.has(String(turn?.status || "")),
  );
  return {
    round: roundIndex + 1,
    outputPath: path.relative(process.cwd(), outputPath),
    status: evidence?.status || "unknown",
    sessionId,
    threadId,
    sessionStatus: canonicalThread.status ?? null,
    threadCount: threads.length,
    sessionThreadCount: threads.filter(
      (thread) => thread?.sessionId === sessionId,
    ).length,
    turnCount: turns.length,
    itemCount: items.length,
    terminalTurnCount: terminalTurns.length,
    turnIds,
    itemIds,
    turnStatusCounts: countByStatus(turns),
    itemStatusCounts: countByStatus(items),
    itemKindCounts: countByKind(items),
    requestTimings,
    process: processSnapshot,
    assertions: {
      roundEvidencePassed: evidence?.status === "pass",
      canonicalSessionIdentityPresent: Boolean(sessionId && threadId),
      canonicalThreadListed: containsCanonicalThread(threads, {
        sessionId,
        threadId,
      }),
      exactlyOneTurnRecorded: turns.length === 1,
      terminalTurnRecordedOnce: terminalTurns.length === 1,
      turnIdentityUnique: new Set(turnIds).size === turnIds.length,
      itemIdentityUnique: new Set(itemIds).size === itemIds.length,
      noPendingTurn: turns.every((turn) =>
        TERMINAL_TURN_STATUSES.has(String(turn?.status || "")),
      ),
      processTreeCaptured: processSnapshot.processCount > 0,
      appServerProcessCaptured: processSnapshot.appServerPids.length > 0,
    },
  };
}

export async function collectRestoredSoakRounds({
  evidencePaths,
  page,
  processSnapshot,
  readEvidence,
}) {
  const restored = [];
  for (let roundIndex = 0; roundIndex < evidencePaths.length; roundIndex += 1) {
    restored.push(
      await collectSoakRoundObservation({
        evidence: readEvidence(evidencePaths[roundIndex]),
        outputPath: evidencePaths[roundIndex],
        page,
        processSnapshot,
        roundIndex,
      }),
    );
  }
  return restored;
}

function trendFor(rounds, field) {
  const values = rounds.map((round) => Number(round.process?.[field] || 0));
  return {
    values,
    initialKb: values[0] ?? 0,
    finalKb: values.at(-1) ?? 0,
    deltaKb: (values.at(-1) ?? 0) - (values[0] ?? 0),
    maxKb: values.length > 0 ? Math.max(...values) : 0,
  };
}

export function buildSoakSummary({
  finalShutdown,
  processSnapshots,
  restoredRounds = [],
  restarts,
  rounds,
}) {
  const rssTrend = trendFor(rounds, "totalRssKb");
  const appServerRssTrend = trendFor(rounds, "appServerRssKb");
  const uniqueSessionIds = new Set(rounds.map((round) => round.sessionId));
  const roundDurationsMs = rounds.map((round) => round.durationMs);
  const assertions = {
    allRoundsPassed: rounds.every((round) =>
      Object.values(round.assertions).every(Boolean),
    ),
    roundSessionsIsolated: uniqueSessionIds.size === rounds.length,
    rssGrowthWithinCalibrationBudget: rssTrend.deltaKb <= RSS_GROWTH_BUDGET_KB,
    appServerRssGrowthWithinCalibrationBudget:
      appServerRssTrend.deltaKb <= RSS_GROWTH_BUDGET_KB,
    roundDurationWithinCalibrationBudget: roundDurationsMs.every(
      (durationMs) =>
        Number.isFinite(durationMs) && durationMs <= ROUND_DURATION_BUDGET_MS,
    ),
    coldRestartCountSatisfied: restarts.length >= 2,
    everyElectronProcessReplaced: restarts.every(
      (restart) => restart.electronProcessReplaced === true,
    ),
    everyPreviousProcessTreeExited: restarts.every(
      (restart) => restart.previousProcessTreeExit?.exited === true,
    ),
    readModelsStableAcrossColdRestarts:
      restoredRounds.length === rounds.length &&
      rounds.every((round, index) => {
        const restored = restoredRounds[index];
        return (
          restored?.sessionId === round.sessionId &&
          restored?.threadId === round.threadId &&
          JSON.stringify(restored?.turnIds) === JSON.stringify(round.turnIds) &&
          JSON.stringify(restored?.itemIds) === JSON.stringify(round.itemIds) &&
          JSON.stringify(restored?.turnStatusCounts) ===
            JSON.stringify(round.turnStatusCounts) &&
          JSON.stringify(restored?.itemStatusCounts) ===
            JSON.stringify(round.itemStatusCounts) &&
          JSON.stringify(restored?.itemKindCounts) ===
            JSON.stringify(round.itemKindCounts)
        );
      }),
    finalProcessTreeExited: finalShutdown?.exited === true,
  };
  return {
    schemaVersion: "soak-01.v1",
    claimBoundary:
      "same managed Electron/App Server lifecycle across controlled current-runtime turns, followed by cold restarts with the same isolated user data; localhost fixture, not live-provider or release-duration proof",
    roundCount: rounds.length,
    restartCount: restarts.length,
    rssGrowthBudgetKb: RSS_GROWTH_BUDGET_KB,
    roundDurationBudgetMs: ROUND_DURATION_BUDGET_MS,
    roundDurationsMs,
    rssTrend,
    appServerRssTrend,
    rounds,
    restoredRounds,
    restarts,
    processSnapshots,
    finalShutdown,
    assertions,
  };
}
