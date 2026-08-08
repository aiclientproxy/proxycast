#!/usr/bin/env node

import process from "node:process";

import { buildVitestLayerReport } from "./lib/vitest-layer-report.mjs";

const DEFAULT_MAX_COMPONENT_CANDIDATES = 37;

export function parseArgs(argv) {
  const options = {
    json: false,
    maxComponentCandidates: DEFAULT_MAX_COMPONENT_CANDIDATES,
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--json") {
      options.json = true;
    } else if (arg === "--max-component-candidates") {
      const value = argv[index + 1];
      if (value === undefined) {
        throw new Error("--max-component-candidates requires a value");
      }
      options.maxComponentCandidates = parseNonNegativeInteger(
        value,
        "--max-component-candidates",
      );
      index += 1;
    } else if (arg.startsWith("--max-component-candidates=")) {
      options.maxComponentCandidates = parseNonNegativeInteger(
        arg.slice("--max-component-candidates=".length),
        "--max-component-candidates",
      );
    } else if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else {
      throw new Error(`Unknown option: ${arg}`);
    }
  }

  return options;
}

function parseNonNegativeInteger(value, label) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed < 0 || String(parsed) !== value) {
    throw new Error(`${label} must be a non-negative integer`);
  }
  return parsed;
}

export function evaluateVitestLayerBudget(report, options = {}) {
  const maxComponentCandidates =
    options.maxComponentCandidates ?? DEFAULT_MAX_COMPONENT_CANDIDATES;
  const candidates = report.componentUnitMigrationCandidates ?? {
    total: 0,
    files: [],
  };
  const overBudget = candidates.total > maxComponentCandidates;

  return {
    ok: !overBudget,
    maxComponentCandidates,
    componentCandidates: candidates.total,
    overBudgetBy: overBudget ? candidates.total - maxComponentCandidates : 0,
    candidateFiles: candidates.files ?? [],
  };
}

export function renderBudgetResultText(result) {
  const lines = [
    "Vitest layer budget check",
    `Component unit-migration candidates: ${result.componentCandidates}`,
    `Budget: ${result.maxComponentCandidates}`,
    `Status: ${result.ok ? "ok" : "over budget"}`,
  ];

  if (!result.ok) {
    lines.push(`Over budget by: ${result.overBudgetBy}`);
    if (result.candidateFiles.length > 0) {
      lines.push("Candidate files:");
      for (const candidate of result.candidateFiles.slice(0, 10)) {
        lines.push(`- ${candidate.file} (${candidate.hints.join(", ")})`);
      }
      if (result.candidateFiles.length > 10) {
        lines.push(`- ... ${result.candidateFiles.length - 10} more`);
      }
    }
  }

  return `${lines.join("\n")}\n`;
}

function printHelp() {
  console.log(`Usage:
  node scripts/check-vitest-layer-budget.mjs [options]

Options:
  --max-component-candidates N  允许的 component VM 迁移候选上限，默认 ${DEFAULT_MAX_COMPONENT_CANDIDATES}
  --json                        输出 JSON
  --help                        显示帮助
`);
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }

  const report = buildVitestLayerReport();
  const result = evaluateVitestLayerBudget(report, options);

  if (options.json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    process.stdout.write(renderBudgetResultText(result));
  }

  if (!result.ok) {
    process.exitCode = 1;
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
