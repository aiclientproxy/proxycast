#!/usr/bin/env node

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { normalizeVersion } from "./windows-squirrel-rc-smoke.mjs";
import { normalizeCandidateSha } from "./lib/release-candidate-identity.mjs";

const SCENARIO_ID = "PLT-02-windows-packaged-evidence";
const RESOURCE_MANIFEST = "desktop-resources.manifest.json";
const APPLICATION_ID = "com.limecloud.lime";
const REQUIRED_RESOURCES = {
  "app-server": "app-server/win32-x64/app-server.exe",
  "code-mode-host": "app-server/win32-x64/code-mode-host.exe",
  "windows-sandbox-setup": "app-server/win32-x64/windows-sandbox-setup.exe",
  "windows-sandbox-runner": "app-server/win32-x64/windows-sandbox-runner.exe",
  "windows-native-host": "native/windows/windows-native-host.exe",
};

export function parseArgs(argv) {
  const options = {
    version: null,
    candidateSha: null,
    squirrelSummary: null,
    codeModeSummary: null,
    nativeHostSummary: null,
    output: path.resolve(
      ".lime/qc/gui-evidence/windows-packaged-evidence/summary.json",
    ),
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "-h" || arg === "--help") {
      options.help = true;
      continue;
    }
    const value = argv[index + 1];
    const optionKey = {
      "--version": "version",
      "--candidate-sha": "candidateSha",
      "--squirrel-summary": "squirrelSummary",
      "--code-mode-summary": "codeModeSummary",
      "--native-host-summary": "nativeHostSummary",
      "--output": "output",
    }[arg];
    if (optionKey && value && !value.startsWith("--")) {
      options[optionKey] =
        arg === "--version" || arg === "--candidate-sha"
          ? value
          : path.resolve(value);
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument or missing value: ${arg}`);
  }
  if (options.help) return options;
  if (!options.version) throw new Error("--version is required");
  if (!options.candidateSha) throw new Error("--candidate-sha is required");
  if (!options.squirrelSummary) {
    throw new Error("--squirrel-summary is required");
  }
  if (!options.codeModeSummary) {
    throw new Error("--code-mode-summary is required");
  }
  if (!options.nativeHostSummary) {
    throw new Error("--native-host-summary is required");
  }
  options.version = normalizeVersion(options.version);
  options.candidateSha = normalizeCandidateSha(options.candidateSha);
  return options;
}

export function buildWindowsPackagedEvidenceSummary({
  version,
  candidateSha,
  squirrelSummary,
  codeModeSummary,
  nativeHostSummary,
  sources = {},
  fileExists = fs.existsSync,
  readFile = (filePath) => fs.readFileSync(filePath),
}) {
  const candidateVersion = normalizeVersion(version);
  const normalizedCandidateSha = normalizeCandidateSha(candidateSha);
  const checks = [];
  const failures = [];
  const check = (name, assertion) => {
    try {
      const detail = assertion();
      checks.push({ name, status: "passed", ...(detail || {}) });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      checks.push({ name, status: "failed", error: message });
      failures.push({ name, error: message });
    }
  };

  let installation = null;
  let resourcesRoot = null;
  let resourceManifest = null;
  check("squirrel-summary", () => {
    requireObject(squirrelSummary, "Windows Squirrel summary");
    requireEqual(
      squirrelSummary.scenarioId,
      "PLT-02-windows-squirrel-rc",
      "Squirrel scenarioId",
    );
    requireEqual(squirrelSummary.result, "pass", "Squirrel result");
    requireEqual(squirrelSummary.platform?.os, "win32", "Squirrel platform.os");
    requireEqual(
      squirrelSummary.platform?.arch,
      "x64",
      "Squirrel platform.arch",
    );
    requireEqual(
      squirrelSummary.platform?.appVersion,
      candidateVersion,
      "Squirrel platform.appVersion",
    );
    requireEqual(
      squirrelSummary.candidateSha,
      normalizedCandidateSha,
      "Squirrel candidateSha",
      true,
    );
    if (!String(squirrelSummary.candidateRunId || "").trim()) {
      throw new Error("Squirrel candidateRunId is missing");
    }
    const squirrelAssertions = squirrelSummary.assertions?.details;
    requireObject(squirrelAssertions, "Squirrel assertions.details");
    for (const name of [
      "uninstallExitZero",
      "uninstalledAppDirectoryRemoved",
      "uninstalledExecutableAbsent",
      "shortcutsRemoved",
    ]) {
      requireEqual(squirrelAssertions[name], true, `Squirrel ${name}`);
    }
    requireObject(
      squirrelSummary.evidence?.uninstall,
      "Squirrel uninstall evidence",
    );
    installation = squirrelSummary.evidence?.installation;
    requireObject(installation, "Squirrel installation evidence");
    requireAbsolutePath(installation.executable, "installed executable");
    requireAbsolutePath(installation.appDirectory, "installed app directory");
    requireAbsolutePath(installation.packageRoot, "installed package root");
    requireAbsolutePath(installation.updateExecutable, "installed Update.exe");
    requireEqual(
      basenamePortable(installation.executable),
      "lime.exe",
      "installed executable basename",
      true,
    );
    requireEqual(
      normalizePortablePath(installation.executable),
      normalizePortablePath(
        joinPortable(installation.appDirectory, "Lime.exe"),
      ),
      "installed executable/app directory",
    );
    requireEqual(
      normalizePortablePath(installation.appDirectory),
      normalizePortablePath(
        joinPortable(installation.packageRoot, `app-${candidateVersion}`),
      ),
      "installed app directory/version",
    );
    if (!fileExists(installation.executable)) {
      throw new Error(
        `installed executable is missing: ${installation.executable}`,
      );
    }
    resourcesRoot = joinPortable(
      dirnamePortable(installation.executable),
      "resources",
    );
    if (!fileExists(joinPortable(resourcesRoot, RESOURCE_MANIFEST))) {
      throw new Error(
        `installed resource manifest is missing: ${joinPortable(resourcesRoot, RESOURCE_MANIFEST)}`,
      );
    }
    resourceManifest = JSON.parse(
      readFile(joinPortable(resourcesRoot, RESOURCE_MANIFEST)),
    );
    return {
      candidateRunId: squirrelSummary.candidateRunId,
      executable: installation.executable,
      resourcesRoot,
    };
  });

  check("installed-resource-manifest", () => {
    requireObject(resourceManifest, "installed desktop resource manifest");
    requireEqual(resourceManifest.schemaVersion, 1, "resource schemaVersion");
    requireEqual(
      resourceManifest.applicationId,
      APPLICATION_ID,
      "resource applicationId",
    );
    requireEqual(resourceManifest.platform, "win32", "resource platform");
    requireEqual(
      resourceManifest.version,
      candidateVersion,
      "resource version",
    );
    requireEqual(resourceManifest.arch, "x64", "resource arch");
    requireEqual(
      resourceManifest.platformKey,
      "win32-x64",
      "resource platformKey",
    );
    const resourceEntries = Array.isArray(resourceManifest.resources)
      ? resourceManifest.resources
      : [];
    const resources = new Map(
      resourceEntries.map((resource) => [resource?.id, resource]),
    );
    if (resources.size !== resourceEntries.length) {
      throw new Error("installed resource manifest contains duplicate ids");
    }
    for (const [id, expectedPath] of Object.entries(REQUIRED_RESOURCES)) {
      const resource = resources.get(id);
      requireObject(resource, `resource ${id}`);
      requireEqual(
        normalizePortablePath(resource.path),
        normalizePortablePath(expectedPath),
        `resource ${id} path`,
      );
      const absolutePath = joinPortable(resourcesRoot, expectedPath);
      if (!fileExists(absolutePath)) {
        throw new Error(`resource ${id} is missing: ${absolutePath}`);
      }
      requireSha256(resource.sha256, `resource ${id} sha256`);
      const actualSha256 = createHash("sha256")
        .update(readFile(absolutePath))
        .digest("hex");
      requireEqual(
        actualSha256,
        resource.sha256,
        `resource ${id} sha256`,
        true,
      );
    }
    return { resourceIds: [...resources.keys()].filter(Boolean).sort() };
  });

  const candidateRunId = squirrelSummary?.candidateRunId || null;
  check("code-mode-summary", () => {
    requireObject(codeModeSummary, "CodeMode summary");
    requireEqual(codeModeSummary.status, "pass", "CodeMode result");
    requireEqual(
      codeModeSummary.packagedExecutable,
      true,
      "CodeMode packagedExecutable",
    );
    requireEqual(
      codeModeSummary.candidateRunId,
      candidateRunId,
      "CodeMode candidateRunId",
    );
    requireEqual(
      codeModeSummary.candidateSha,
      normalizedCandidateSha,
      "CodeMode candidateSha",
      true,
    );
    requireEqual(
      normalizePortablePath(codeModeSummary.packagedExecutablePath),
      normalizePortablePath(installation?.executable),
      "CodeMode packaged executable",
    );
    requirePackagedProcess(
      codeModeSummary.processes?.appServerCommand,
      joinPortable(resourcesRoot, REQUIRED_RESOURCES["app-server"]),
      "CodeMode app-server process",
    );
    requirePackagedProcess(
      codeModeSummary.processes?.codeModeHostCommand,
      joinPortable(resourcesRoot, REQUIRED_RESOURCES["code-mode-host"]),
      "CodeMode code-mode-host process",
    );
    requireEqual(
      codeModeSummary.processes?.codeModeHostParentPid,
      codeModeSummary.processes?.appServerPid,
      "CodeMode process ownership",
    );
  });

  check("native-host-summary", () => {
    requireObject(nativeHostSummary, "Windows native host summary");
    requireEqual(nativeHostSummary.result, "passed", "native host result");
    requireEqual(
      nativeHostSummary.evidenceLevel,
      "gate-b",
      "native host evidenceLevel",
    );
    requireEqual(nativeHostSummary.platform, "win32", "native host platform");
    requireEqual(nativeHostSummary.arch, "x64", "native host arch");
    requireEqual(
      nativeHostSummary.candidateRunId,
      candidateRunId,
      "native host candidateRunId",
    );
    requireEqual(
      nativeHostSummary.candidateSha,
      normalizedCandidateSha,
      "native host candidateSha",
      true,
    );
    requireEqual(
      normalizePortablePath(nativeHostSummary.electronExecutable),
      normalizePortablePath(installation?.executable),
      "native host Electron executable",
    );
    requireEqual(
      nativeHostSummary.helper?.readOnly,
      true,
      "native host readOnly",
    );
    requireEqual(
      nativeHostSummary.helper?.digestMatches,
      true,
      "native host digest",
    );
    requireEqual(
      normalizePortablePath(nativeHostSummary.helper?.resourcesRoot),
      normalizePortablePath(resourcesRoot),
      "native host resources root",
    );
    requireEqual(
      normalizePortablePath(nativeHostSummary.helper?.path),
      normalizePortablePath(
        joinPortable(resourcesRoot, REQUIRED_RESOURCES["windows-native-host"]),
      ),
      "native host helper path",
    );
  });

  return {
    schemaVersion: 1,
    scenarioId: SCENARIO_ID,
    proofLevel: "Gate B",
    claimBoundary:
      "The Windows Squirrel-installed Lime.exe, packaged app-server/code-mode-host processes, installed desktop resource manifest, and read-only native host Gate B all refer to one candidate installation; this does not replace real Windows GUI visual comparison with Codex Desktop.",
    result: failures.length === 0 ? "passed" : "failed",
    candidate: {
      version: candidateVersion,
      sha: normalizedCandidateSha,
      runId: candidateRunId,
      executable: installation?.executable || null,
      resourcesRoot,
    },
    checks,
    ...(failures.length > 0 ? { failures } : {}),
    sources,
  };
}

export function validateWindowsPackagedEvidence({
  version,
  candidateSha,
  squirrelSummary,
  codeModeSummary,
  nativeHostSummary,
  sources,
  fileExists,
  readFile,
}) {
  return buildWindowsPackagedEvidenceSummary({
    version,
    candidateSha,
    squirrelSummary,
    codeModeSummary,
    nativeHostSummary,
    sources,
    fileExists,
    readFile,
  });
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function requireObject(value, label) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} is missing`);
  }
}

function requireEqual(actual, expected, label, caseInsensitive = false) {
  const left = caseInsensitive ? String(actual).toLowerCase() : actual;
  const right = caseInsensitive ? String(expected).toLowerCase() : expected;
  if (left !== right) {
    throw new Error(
      `${label} mismatch: expected ${String(expected)}, got ${String(actual)}`,
    );
  }
}

function requireAbsolutePath(value, label) {
  if (!isAbsolutePortable(value)) {
    throw new Error(`${label} must be absolute: ${String(value)}`);
  }
}

function requireSha256(value, label) {
  if (!/^[a-f0-9]{64}$/iu.test(String(value || ""))) {
    throw new Error(`${label} is invalid`);
  }
}

function requirePackagedProcess(command, expectedPath, label) {
  const normalizedCommand = normalizePortablePath(command);
  const normalizedExpected = normalizePortablePath(expectedPath);
  if (!normalizedCommand || !normalizedCommand.includes(normalizedExpected)) {
    throw new Error(
      `${label} does not point to installed resources: ${String(command)}`,
    );
  }
}

function isAbsolutePortable(value) {
  return /^(?:[a-z]:[\\/]|\\\\|\/)/iu.test(String(value || "").trim());
}

function normalizePortablePath(value) {
  return String(value || "")
    .trim()
    .replace(/^"|"$/g, "")
    .replaceAll("\\", "/")
    .replace(/\/+/g, "/")
    .replace(/\/$/, "")
    .toLowerCase();
}

function basenamePortable(value) {
  const normalized = normalizePortablePath(value);
  return normalized.slice(normalized.lastIndexOf("/") + 1);
}

function dirnamePortable(value) {
  const raw = String(value || "")
    .trim()
    .replace(/^"|"$/g, "");
  const separator = Math.max(raw.lastIndexOf("/"), raw.lastIndexOf("\\"));
  return separator > 0 ? raw.slice(0, separator) : raw;
}

function joinPortable(base, relative) {
  return path.join(
    String(base || ""),
    ...String(relative || "").split(/[\\/]+/u),
  );
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
  } catch (error) {
    const output = resolveOutputArg(
      process.argv.slice(2),
      path.resolve(
        ".lime/qc/gui-evidence/windows-packaged-evidence/summary.json",
      ),
    );
    const summary = {
      schemaVersion: 1,
      scenarioId: SCENARIO_ID,
      proofLevel: "Gate B",
      result: "failed",
      candidate: {
        version: null,
        sha: null,
        runId: null,
        executable: null,
        resourcesRoot: null,
      },
      checks: [],
      failures: [
        {
          name: "arguments",
          error: error instanceof Error ? error.message : String(error),
        },
      ],
      sources: {},
    };
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, `${JSON.stringify(summary, null, 2)}\n`);
    console.error(
      `[windows-packaged-evidence] result=failed summary=${output}`,
    );
    process.exitCode = 1;
    return;
  }
  if (options.help) {
    console.log(
      "Usage: node scripts/electron/windows-packaged-evidence.mjs --version <version> --candidate-sha <sha> --squirrel-summary <path> --code-mode-summary <path> --native-host-summary <path> --output <path>",
    );
    return;
  }
  let summary;
  try {
    summary = validateWindowsPackagedEvidence({
      version: options.version,
      candidateSha: options.candidateSha,
      squirrelSummary: readJson(options.squirrelSummary),
      codeModeSummary: readJson(options.codeModeSummary),
      nativeHostSummary: readJson(options.nativeHostSummary),
      sources: {
        squirrelSummary: options.squirrelSummary,
        codeModeSummary: options.codeModeSummary,
        nativeHostSummary: options.nativeHostSummary,
      },
    });
  } catch (error) {
    summary = {
      schemaVersion: 1,
      scenarioId: SCENARIO_ID,
      proofLevel: "Gate B",
      result: "failed",
      candidate: {
        version: options.version,
        sha: options.candidateSha,
        runId: null,
        executable: null,
        resourcesRoot: null,
      },
      checks: [],
      failures: [
        {
          name: "input",
          error: error instanceof Error ? error.message : String(error),
        },
      ],
      sources: {
        squirrelSummary: options.squirrelSummary,
        codeModeSummary: options.codeModeSummary,
        nativeHostSummary: options.nativeHostSummary,
      },
    };
  }
  fs.mkdirSync(path.dirname(options.output), { recursive: true });
  fs.writeFileSync(options.output, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(
    `[windows-packaged-evidence] result=${summary.result} summary=${options.output}`,
  );
  if (summary.result !== "passed") process.exitCode = 1;
}

function resolveOutputArg(argv, fallback) {
  const outputIndex = argv.indexOf("--output");
  const output = outputIndex >= 0 ? argv[outputIndex + 1] : null;
  return output && !output.startsWith("--") ? path.resolve(output) : fallback;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error) => {
    console.error(`[windows-packaged-evidence] failed: ${error.message}`);
    process.exitCode = 1;
  });
}
