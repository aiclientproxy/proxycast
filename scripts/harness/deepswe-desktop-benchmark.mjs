#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

import {
  DEEPSWE_DESKTOP_TRIAL_SCHEMA,
  evaluateDesktopSuite,
  loadDesktopManifest,
  preflightDesktopManifest,
} from "./deepswe-desktop-contract.mjs";

const DEFAULT_OUTPUT = ".lime/benchmark/v2/desktop/summary.json";

function usage() {
  return `
DeepSWE Desktop Benchmark

Usage:
  node scripts/harness/deepswe-desktop-benchmark.mjs --preflight
  node scripts/harness/deepswe-desktop-benchmark.mjs --evidence <file-or-directory> [options]

Options:
  --preflight             Validate Desktop Smoke 5 source/task contract
  --evidence <path>       Trial JSON file or directory containing trial JSON files
  --output <path>         Suite summary path, default ${DEFAULT_OUTPUT}
  --no-write              Print summary without writing it
  -h, --help              Show this help
`;
}

export function parseArgs(argv) {
  const options = {
    preflight: false,
    evidencePath: null,
    output: path.resolve(DEFAULT_OUTPUT),
    write: true,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--preflight") {
      options.preflight = true;
      continue;
    }
    if (arg === "--evidence" && argv[index + 1]) {
      options.evidencePath = path.resolve(String(argv[index + 1]));
      index += 1;
      continue;
    }
    if (arg === "--output" && argv[index + 1]) {
      options.output = path.resolve(String(argv[index + 1]));
      index += 1;
      continue;
    }
    if (arg === "--no-write") {
      options.write = false;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }
  if (!options.help && !options.preflight && !options.evidencePath) {
    throw new Error("either --preflight or --evidence is required");
  }
  return options;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function collectJsonFiles(rootPath) {
  const stat = fs.statSync(rootPath);
  if (stat.isFile()) return [rootPath];
  if (!stat.isDirectory()) {
    throw new Error(`evidence path is not a file or directory: ${rootPath}`);
  }
  const files = [];
  const stack = [rootPath];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const candidate = path.join(current, entry.name);
      if (entry.isDirectory()) stack.push(candidate);
      if (entry.isFile() && entry.name.endsWith(".json")) files.push(candidate);
    }
  }
  return files.sort();
}

export function readTrialEvidence(evidencePath) {
  return collectJsonFiles(evidencePath)
    .map((filePath) => ({ filePath, value: readJson(filePath) }))
    .filter(
      ({ value }) => value?.schemaVersion === DEEPSWE_DESKTOP_TRIAL_SCHEMA,
    )
    .map(({ filePath, value }) => ({
      ...value,
      evidencePath: value.evidencePath || filePath,
    }));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

export function runBenchmark(options, repoRoot = process.cwd()) {
  const { manifest } = loadDesktopManifest(repoRoot);
  if (options.preflight) {
    const preflight = preflightDesktopManifest({ repoRoot, manifest });
    if (preflight.status !== "pass") process.exitCode = 1;
    return preflight;
  }
  const evidenceList = readTrialEvidence(options.evidencePath);
  if (evidenceList.length === 0) {
    throw new Error(
      `no ${DEEPSWE_DESKTOP_TRIAL_SCHEMA} evidence found: ${options.evidencePath}`,
    );
  }
  const summary = {
    ...evaluateDesktopSuite({ evidenceList, manifest, repoRoot }),
    generatedAt: new Date().toISOString(),
    sourceCommit: manifest.sourceCommit,
    evidenceRoot: options.evidencePath,
  };
  if (options.write) writeJson(options.output, summary);
  if (!summary.desktopCodingPass) process.exitCode = 2;
  return summary;
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(usage());
    return;
  }
  const result = runBenchmark(options);
  console.log(JSON.stringify(result, null, 2));
}

if (
  process.argv[1] &&
  pathToFileURL(process.argv[1]).href === import.meta.url
) {
  main().catch((error) => {
    console.error(
      error instanceof Error ? error.stack || error.message : error,
    );
    process.exitCode = 1;
  });
}
