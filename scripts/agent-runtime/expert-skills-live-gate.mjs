#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const DEFAULT_DETERMINISTIC_SUMMARY =
  ".lime/qc/gui-evidence/claw-chat-current-fixture/claw-chat-current-fixture-expert-panel-skills-runtime-regression-summary.json";

const CORE_EXPERT_ASSERTIONS = [
  "expertSkillsRuntimePromptReachedBackend",
  "expertSkillsRuntimeMetadataReachedBackend",
  "expertDeclaredSkillRefsObserved",
  "expertSelectedSkillObserved",
  "expertInvokedSkillObserved",
  "readModelExpertSkillsRuntimeCompleted",
  "readModelExpertSkillSearchObserved",
  "readModelExpertSkillInvocationObserved",
  "readModelExpertSkillBodyReadObserved",
  "readModelExpertSkillGateObserved",
  "expertSkillSearchBeforeSkillInvocation",
];

const PANEL_EXPERT_ASSERTIONS = [
  "expertPanelSecondTurnPromptReachedBackend",
  "expertPanelSkillRefsOverrideReachedBackend",
  "expertPanelReadModelCompleted",
  "expertPanelReadModelSkillBodyReadObserved",
  "expertPanelReadModelSkillGateObserved",
  "expertPanelReadModelSkillSearchObserved",
  "expertPanelReadModelSkillInvocationObserved",
  "expertPanelSkillSearchBeforeSkillInvocation",
];

function isRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value);
}

function readJsonFile(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function boolAt(value, keys) {
  let cursor = value;
  for (const key of keys) {
    if (!isRecord(cursor)) {
      return false;
    }
    cursor = cursor[key];
  }
  return cursor === true;
}

function collectAssertions(summary) {
  return {
    ...(isRecord(summary?.commonAssertions) ? summary.commonAssertions : {}),
    ...(isRecord(summary?.scenarioAssertions)
      ? summary.scenarioAssertions
      : {}),
    ...(isRecord(summary?.assertions) ? summary.assertions : {}),
  };
}

function missingTrueKeys(assertions, keys) {
  return keys.filter((key) => assertions[key] !== true);
}

function expertReadModel(summary) {
  return (
    summary?.readModelExpertPanelSkillsRuntime ??
    summary?.readModelExpertSkillsRuntime ??
    null
  );
}

function readModelIssues(summary) {
  const readModel = expertReadModel(summary);
  if (!isRecord(readModel)) {
    return ["missing expert skills canonical read-model summary"];
  }
  const issues = [];
  if (readModel.hasThreadRead !== true) {
    issues.push("canonical read model missing");
  }
  if (
    !Number.isFinite(readModel.skillSearchCount) ||
    readModel.skillSearchCount < 1
  ) {
    issues.push("skill_search summary missing");
  }
  if (
    !Number.isFinite(readModel.skillInvocationCount) ||
    readModel.skillInvocationCount < 1
  ) {
    issues.push("Skill invocation summary missing");
  }
  for (const [key, label] of [
    ["skillBodyReadObserved", "SKILL.md body read missing"],
    ["skillGateObserved", "Skill gate missing"],
    ["expertDeclaredObserved", "expert declared skillRefs missing"],
    ["expertSelectedObserved", "expert selected skill missing"],
    ["expertInvokedObserved", "expert invoked skill missing"],
    ["skillSearchBeforeSkillInvocation", "skill_search ordering missing"],
  ]) {
    if (readModel[key] !== true) {
      issues.push(label);
    }
  }
  return issues;
}

function hasLiveProviderStatement(summary) {
  return (
    summary?.liveProviderUsed === true ||
    boolAt(summary, ["assertions", "liveProviderUsed"]) ||
    boolAt(summary, ["commonAssertions", "liveProviderUsed"]) ||
    boolAt(summary, ["scenarioAssertions", "liveProviderUsed"]) ||
    boolAt(summary, ["liveProvider", "used"])
  );
}

function isFixtureProvider(summary) {
  return (
    summary?.provider === "fixture-provider" ||
    summary?.model === "fixture-model"
  );
}

function evaluateSummary(summary, { requireLiveProvider }) {
  const assertions = collectAssertions(summary);
  const requiredAssertions =
    summary?.scenario === "expert-panel-skills-runtime"
      ? PANEL_EXPERT_ASSERTIONS
      : CORE_EXPERT_ASSERTIONS;

  const issues = [];
  if (summary?.ok !== true) {
    issues.push("summary.ok is not true");
  }
  issues.push(
    ...missingTrueKeys(assertions, requiredAssertions).map(
      (key) => `missing assertion ${key}`,
    ),
  );
  issues.push(...readModelIssues(summary));

  if (requireLiveProvider) {
    if (!hasLiveProviderStatement(summary)) {
      issues.push("liveProviderUsed=true statement missing");
    }
    if (isFixtureProvider(summary)) {
      issues.push("summary still uses fixture provider/model");
    }
  } else if (assertions.liveProviderNotUsed !== true) {
    issues.push("deterministic summary must state liveProviderNotUsed=true");
  }

  return {
    status: issues.length === 0 ? "pass" : "fail",
    issues,
    scenario: summary?.scenario ?? null,
    provider: summary?.provider ?? null,
    model: summary?.model ?? null,
    readModel: expertReadModel(summary),
  };
}

function resolvePath(repoRoot, inputPath) {
  return path.isAbsolute(inputPath)
    ? inputPath
    : path.join(repoRoot, inputPath);
}

function readOptionalSummary(repoRoot, inputPath) {
  if (!inputPath) {
    return null;
  }
  const absolutePath = resolvePath(repoRoot, inputPath);
  if (!fs.existsSync(absolutePath)) {
    return {
      status: "missing",
      path: absolutePath,
      issues: [`summary not found: ${absolutePath}`],
    };
  }
  return {
    status: "found",
    path: absolutePath,
    summary: readJsonFile(absolutePath),
  };
}

export function buildExpertSkillsLiveGateReport(options = {}) {
  const repoRoot = options.repoRoot ?? process.cwd();
  const deterministicPath =
    options.deterministicSummary ?? DEFAULT_DETERMINISTIC_SUMMARY;
  const deterministicInput = readOptionalSummary(repoRoot, deterministicPath);
  const deterministic =
    deterministicInput?.status === "found"
      ? {
          path: deterministicInput.path,
          ...evaluateSummary(deterministicInput.summary, {
            requireLiveProvider: false,
          }),
        }
      : {
          path:
            deterministicInput?.path ??
            resolvePath(repoRoot, deterministicPath),
          status: "missing",
          issues: deterministicInput?.issues ?? [
            "deterministic summary missing",
          ],
        };

  const liveInput = readOptionalSummary(repoRoot, options.liveSummary);
  const live =
    liveInput?.status === "found"
      ? {
          path: liveInput.path,
          ...evaluateSummary(liveInput.summary, {
            requireLiveProvider: true,
          }),
        }
      : {
          path: liveInput?.path ?? null,
          status: "missing",
          issues: liveInput?.issues ?? [
            "live Provider expert skills summary missing",
          ],
        };

  const deterministicReady = deterministic.status === "pass";
  const liveReady = live.status === "pass";
  const status =
    deterministicReady && liveReady
      ? "pass"
      : deterministicReady &&
          options.allowMissingLive &&
          live.status === "missing"
        ? "pending_live_provider"
        : "fail";

  return {
    status,
    ok: status === "pass" || status === "pending_live_provider",
    deterministic,
    live,
    completion: {
      deterministicExpertSkillsReady: deterministicReady,
      liveProviderExpertSkillsReady: liveReady,
      overallGoalReady: deterministicReady && liveReady,
    },
    nextRequired:
      deterministicReady && !liveReady
        ? "run explicit live Provider expert skills validation and pass its summary with --live-summary"
        : null,
  };
}

function parseArgs(argv) {
  const options = {
    deterministicSummary: DEFAULT_DETERMINISTIC_SUMMARY,
    liveSummary: "",
    allowMissingLive: false,
    format: "text",
    output: "",
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const next = argv[index + 1];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    if (arg === "--deterministic-summary" && next) {
      options.deterministicSummary = next;
      index += 1;
      continue;
    }
    if (arg === "--live-summary" && next) {
      options.liveSummary = next;
      index += 1;
      continue;
    }
    if (arg === "--allow-missing-live") {
      options.allowMissingLive = true;
      continue;
    }
    if (arg === "--format" && next) {
      options.format = next;
      index += 1;
      continue;
    }
    if (arg === "--output" && next) {
      options.output = next;
      index += 1;
      continue;
    }
    throw new Error(`未知参数: ${arg}`);
  }
  return options;
}

function printHelp() {
  console.log(`
Expert Skills Live Gate

用途:
  审计专家 Skills Runtime 的确定性 Electron 证据和显式 live Provider 证据。
  本脚本只读取 summary artifact，不会调用真实模型。

用法:
  node scripts/agent-runtime/expert-skills-live-gate.mjs
  node scripts/agent-runtime/expert-skills-live-gate.mjs --allow-missing-live
  node scripts/agent-runtime/expert-skills-live-gate.mjs --live-summary .lime/qc/.../summary.json

选项:
  --deterministic-summary <path>  确定性专家 Skills summary，默认 current regression artifact
  --live-summary <path>           显式 live Provider 专家 Skills summary
  --allow-missing-live            live 缺失时返回 pending 而非失败
  --format json|text              输出格式
  --output <path>                 写出 report JSON
`);
}

function printTextReport(report) {
  console.log(`EXPERT_SKILLS_LIVE_GATE_RESULT=${report.status}`);
  console.log(
    `deterministic=${report.deterministic.status} live=${report.live.status}`,
  );
  if (report.nextRequired) {
    console.log(`next=${report.nextRequired}`);
  }
  for (const [section, result] of [
    ["deterministic", report.deterministic],
    ["live", report.live],
  ]) {
    for (const issue of result.issues ?? []) {
      console.log(`${section}: ${issue}`);
    }
  }
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printHelp();
    return;
  }
  const report = buildExpertSkillsLiveGateReport(options);
  if (options.output) {
    const outputPath = resolvePath(process.cwd(), options.output);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  if (options.format === "json") {
    console.log(JSON.stringify(report, null, 2));
  } else {
    printTextReport(report);
  }
  if (report.status === "fail") {
    process.exitCode = 1;
  }
}

const currentFile = fileURLToPath(import.meta.url);
if (process.argv[1] && path.resolve(process.argv[1]) === currentFile) {
  main();
}
