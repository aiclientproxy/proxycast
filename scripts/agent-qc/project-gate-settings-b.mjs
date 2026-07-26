#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import {
  buildSettingsGateBFailureEvidence,
  buildSettingsGateBFEvidence,
  validateSettingsGateBRunId,
} from "../lib/project-gate-settings-b-core.mjs";

function standaloneRunId() {
  const timestamp = new Date().toISOString().replace(/[-:.]/g, "");
  return `standalone-settings-b-${timestamp}-${process.pid}`;
}

function printHelp() {
  console.log(`
Project Gate SETTINGS-01 Gate B-F evidence aggregator

Usage:
  node scripts/agent-qc/project-gate-settings-b.mjs \\
    --run-id <candidate-run-id> \\
    --source shell-memory=<summary.json> \\
    --source settings-scenario=<summary.json>

Options:
  --run-id <id>        Candidate or explicit standalone run-id
  --source <kind>=<path>
                       Same-run owner evidence. Repeat for each scenario.
                       Kinds: shell-memory, settings-scenario
  --evidence-dir <dir> Output directory. Must remain under the run evidence root.
  -h, --help           Show help

The aggregator never relabels an owner summary. It validates the exact owner
claim, binds it to the same run-id, and writes SETTINGS-01 gate-b-f evidence.
`);
}

function parseArgs(argv) {
  const options = {
    runId: process.env.LIME_GATE_RUN_ID?.trim() || null,
    evidenceDir: null,
    sources: [],
    help: false,
  };
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
    if (arg === "--source" && next) {
      const separator = next.indexOf("=");
      if (separator <= 0 || separator === next.length - 1) {
        throw new Error("--source must use <kind>=<summary-path>");
      }
      options.sources.push({
        kind: next.slice(0, separator).trim(),
        path: path.resolve(next.slice(separator + 1).trim()),
      });
      index += 1;
      continue;
    }
    throw new Error(`unknown argument: ${arg}`);
  }
  options.runId = validateSettingsGateBRunId(
    options.runId || standaloneRunId(),
  );
  return options;
}

function isPathInside(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return (
    relative !== "" && !relative.startsWith("..") && !path.isAbsolute(relative)
  );
}

function readSourceRecords({ repoRoot, runId, sources }) {
  const runRoot = path.resolve(repoRoot, ".lime", "qc", "project-gates", runId);
  return sources.map((source) => {
    const sourcePath = fs.realpathSync(source.path);
    if (!isPathInside(runRoot, sourcePath)) {
      throw new Error(
        `source evidence must be under the same run root: ${sourcePath}`,
      );
    }
    const raw = fs.readFileSync(sourcePath);
    return {
      kind: source.kind,
      file: path.relative(runRoot, sourcePath).replaceAll("\\", "/"),
      sha256: crypto.createHash("sha256").update(raw).digest("hex"),
      value: JSON.parse(raw.toString("utf8")),
    };
  });
}

function writeJsonAtomic(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  fs.writeFileSync(
    temporaryPath,
    `${JSON.stringify(value, null, 2)}\n`,
    "utf8",
  );
  fs.renameSync(temporaryPath, filePath);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const startedAt = new Date().toISOString();
  const repoRoot = process.cwd();
  const runRoot = path.resolve(
    repoRoot,
    ".lime",
    "qc",
    "project-gates",
    options.runId,
  );
  const evidenceDir = options.evidenceDir
    ? path.resolve(options.evidenceDir)
    : path.join(runRoot, "settings-01-gate-b-f");
  if (!isPathInside(runRoot, evidenceDir)) {
    throw new Error("--evidence-dir must remain under the candidate run root");
  }
  const summaryPath = path.join(evidenceDir, "summary.json");

  try {
    const sourceRecords = readSourceRecords({
      repoRoot,
      runId: options.runId,
      sources: options.sources,
    });
    const evidence = buildSettingsGateBFEvidence({
      candidateRunId: options.runId,
      startedAt,
      completedAt: new Date().toISOString(),
      sourceRecords,
    });
    writeJsonAtomic(summaryPath, evidence);
    console.log(
      `[agent-qc:project-gate-settings-b] result=pass complete=${evidence.surfaceProof.complete} scenarios=${evidence.coverage.completed}/${evidence.coverage.total} summary=${summaryPath}`,
    );
  } catch (error) {
    const evidence = buildSettingsGateBFailureEvidence({
      candidateRunId: options.runId,
      startedAt,
      completedAt: new Date().toISOString(),
      error,
    });
    writeJsonAtomic(summaryPath, evidence);
    throw error;
  }
}

main().catch((error) => {
  console.error(
    `[agent-qc:project-gate-settings-b] ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
