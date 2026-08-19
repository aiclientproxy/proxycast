#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

function parseArgs(argv) {
  const result = {
    checkTerminal: false,
    dbPath: ".lime/qc/qcloop-isolated-worker-preflight.db",
    format: "summary",
    help: false,
    jobId: "",
    markdownOutputPath: "",
    outputPath: "",
    processOwnerPath: ".lime/qc/gui-process-owner-current.json",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--db" && argv[index + 1]) {
      result.dbPath = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--job-id" && argv[index + 1]) {
      result.jobId = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--process-owner" && argv[index + 1]) {
      result.processOwnerPath = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--format" && argv[index + 1]) {
      result.format = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--output" && argv[index + 1]) {
      result.outputPath = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--markdown-output" && argv[index + 1]) {
      result.markdownOutputPath = String(argv[index + 1]).trim();
      index += 1;
      continue;
    }
    if (arg === "--check-terminal") {
      result.checkTerminal = true;
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      result.help = true;
    }
  }
  return result;
}

function printHelp() {
  console.log(`
Lime Agent QC qcloop DB Lease Snapshot

用法:
  npm run agent-qc:qcloop-db-lease -- --job-id <qcloop-job-id>
  node scripts/agent-qc/qcloop-db-lease.mjs --db ./.lime/qc/qcloop-isolated-worker-preflight.db --job-id <id> --format json --output ./.lime/qc/qcloop-db-lease-current.json

选项:
  --db PATH                  qcloop SQLite DB，默认 .lime/qc/qcloop-isolated-worker-preflight.db
  --job-id ID                qcloop job id，必填
  --process-owner PATH       raw process owner sidecar，默认 .lime/qc/gui-process-owner-current.json
  --format FMT               summary | json，默认 summary
  --output PATH              写入文件；默认 stdout
  --markdown-output PATH     额外写入 Markdown 摘要
  --check-terminal           job 未终态或仍有 running item 时非 0 退出
  -h, --help                 显示帮助
`);
}

function sqlQuote(value) {
  return `'${String(value).replaceAll("'", "''")}'`;
}

function sqliteJson(dbPath, sql) {
  const output = execFileSync("sqlite3", ["-readonly", "-json", dbPath, sql], {
    encoding: "utf8",
  }).trim();
  return output ? JSON.parse(output) : [];
}

function readJsonIfExists(filePath) {
  const resolved = path.resolve(process.cwd(), filePath);
  if (!fs.existsSync(resolved)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(resolved, "utf8"));
}

function extractScenarioId(itemPreview) {
  return (
    String(itemPreview || "").match(/"scenario_id"\s*:\s*"([^"]+)"/)?.[1] ||
    null
  );
}

function createDbLeaseReport({
  dbPath,
  generatedAt = new Date().toISOString(),
  jobId,
  processOwnerPath,
}) {
  if (!jobId) {
    throw new Error("--job-id is required");
  }
  const quotedJobId = sqlQuote(jobId);
  const jobs = sqliteJson(
    dbPath,
    `select id, name, status, created_at, finished_at from batch_jobs where id=${quotedJobId}`,
  );
  const items = sqliteJson(
    dbPath,
    `select id, batch_job_id, status, current_attempt_no, current_qc_no, tokens_used, lock_owner, lock_expires_at, queued_at, last_error, created_at, finished_at, substr(item_value,1,700) as item_preview from batch_items where batch_job_id=${quotedJobId} order by created_at, id`,
  );
  const attempts = sqliteJson(
    dbPath,
    `select attempts.id, attempts.batch_item_id, attempts.attempt_no, attempts.run_no, attempts.attempt_type, attempts.status, attempts.exit_code, attempts.tokens_used, attempts.started_at, attempts.finished_at, length(attempts.stdout) as stdout_len, length(attempts.stderr) as stderr_len, substr(attempts.stdout,1,700) as stdout_preview, substr(attempts.stderr,1,700) as stderr_preview from attempts join batch_items on batch_items.id=attempts.batch_item_id where batch_items.batch_job_id=${quotedJobId} order by attempts.started_at, attempts.id`,
  );
  const activeItem = items.find((item) => item.status === "running") || null;
  const activeScenario = extractScenarioId(activeItem?.item_preview);
  const processOwner = readJsonIfExists(processOwnerPath) || {
    qcloopProcesses: [],
  };
  const processSnapshot = (processOwner.qcloopProcesses || []).filter(
    (entry) => {
      const command = String(entry?.command || "");
      return (
        command.includes(jobId) ||
        command.includes("18080") ||
        (activeScenario ? command.includes(activeScenario) : false)
      );
    },
  );
  const activeAttempt = activeItem
    ? attempts.find(
        (attempt) =>
          attempt.batch_item_id === activeItem.id &&
          attempt.status === "running",
      ) || null
    : null;
  const observedWorker = activeScenario
    ? processSnapshot.find((entry) =>
        String(entry?.command || "").includes(activeScenario),
      ) || null
    : null;
  const observedQcloop =
    processSnapshot.find((entry) =>
      String(entry?.command || "").includes("serve --addr 127.0.0.1:18080"),
    ) || null;
  const terminalStatuses = new Set([
    "completed",
    "failed",
    "cancelled",
    "canceled",
  ]);
  const jobStatus = jobs[0]?.status || "unknown";
  const terminal =
    terminalStatuses.has(String(jobStatus).toLowerCase()) && !activeItem;

  return {
    schemaVersion: "v1",
    generatedAt,
    jobId,
    dbPath,
    processOwnerPath,
    summary: {
      status: jobStatus,
      terminal,
      activeItemId: activeItem?.id || null,
      activeScenario,
      lockOwner: activeItem?.lock_owner || null,
      lockExpiresAt: activeItem?.lock_expires_at || null,
      activeAttemptId: activeAttempt?.id || null,
      activeAttemptStatus: activeAttempt?.status || null,
      stdoutLength: activeAttempt?.stdout_len ?? null,
      stderrLength: activeAttempt?.stderr_len ?? null,
      observedWorkerPid: observedWorker?.pid || null,
      observedQcloopPid: observedQcloop?.pid || null,
      finding: activeItem
        ? "qcloop DB lease, active attempt, and process snapshot still show a running no-output stale GUI owner; no intervention was performed."
        : "No running item observed in this DB snapshot; verify qcloop status sidecar before changing gates.",
    },
    jobs,
    items,
    attempts,
    processSnapshot,
    guardrails: [
      "read-only DB observation only",
      "do not kill / pause / interrupt stale worker without owner confirmation",
      "do not modify qcloop SQLite DB",
      "do not start another full GUI P0 batch while GUI owner is blocked",
      "do not overwrite .lime/qc/agent-qc-evidence.json before a real 8/8 P0 pass",
    ],
  };
}

function renderSummary(report) {
  const summary = report.summary;
  return (
    [
      `status=${summary.status}`,
      `terminal=${summary.terminal}`,
      `activeItem=${summary.activeItemId || "none"}`,
      `activeScenario=${summary.activeScenario || "none"}`,
      `lockOwner=${summary.lockOwner || "none"}`,
      `lockExpiresAt=${summary.lockExpiresAt || "none"}`,
      `activeAttempt=${summary.activeAttemptId || "none"}:${summary.activeAttemptStatus || "none"}`,
      `stdoutStderr=${summary.stdoutLength ?? "null"}/${summary.stderrLength ?? "null"}`,
      `observedWorkerPid=${summary.observedWorkerPid || "none"}`,
      `observedQcloopPid=${summary.observedQcloopPid || "none"}`,
      `finding=${summary.finding}`,
    ].join("\n") + "\n"
  );
}

function renderMarkdown(report) {
  const summary = report.summary;
  return [
    "# qcloop DB lease snapshot",
    "",
    `- Generated at: ${report.generatedAt}`,
    `- Job: ${report.jobId}`,
    `- DB: ${report.dbPath}`,
    `- Status: ${summary.status}`,
    `- Terminal: ${summary.terminal}`,
    `- Active item: ${summary.activeItemId || "none"}`,
    `- Active scenario: ${summary.activeScenario || "none"}`,
    `- Lock owner: ${summary.lockOwner || "none"}`,
    `- Lock expires at: ${summary.lockExpiresAt || "none"}`,
    `- Active attempt: ${summary.activeAttemptId || "none"} / ${summary.activeAttemptStatus || "none"}`,
    `- stdout/stderr length: ${summary.stdoutLength ?? "null"} / ${summary.stderrLength ?? "null"}`,
    `- Observed worker PID: ${summary.observedWorkerPid || "none"}`,
    `- Observed qcloop PID: ${summary.observedQcloopPid || "none"}`,
    "",
    "## Finding",
    "",
    summary.finding,
    "",
    "## Guardrails",
    "",
    ...report.guardrails.map((guardrail) => `- ${guardrail}`),
    "",
  ].join("\n");
}

function writeOutput(outputPath, content) {
  if (!outputPath) {
    process.stdout.write(content);
    return;
  }
  const resolvedOutputPath = path.resolve(process.cwd(), outputPath);
  fs.mkdirSync(path.dirname(resolvedOutputPath), { recursive: true });
  fs.writeFileSync(resolvedOutputPath, content, "utf8");
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const report = createDbLeaseReport(options);
  const content =
    options.format === "json"
      ? `${JSON.stringify(report, null, 2)}\n`
      : renderSummary(report);
  writeOutput(options.outputPath, content);
  if (options.markdownOutputPath) {
    writeOutput(options.markdownOutputPath, renderMarkdown(report));
  }
  if (options.checkTerminal && !report.summary.terminal) {
    process.exit(1);
  }
}

main();
