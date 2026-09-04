#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const EVIDENCE_INDEX_PATH =
  "internal/exec-plans/codex-desktop-selective-goose-reference-evidence.json";

const PRODUCT_CHAIN =
  "Electron Desktop Host -> App Server JSON-RPC -> RuntimeCore -> Thread/Turn/Item -> GUI";
const SOURCE_ROOTS = ["electron", "src", "packages", "lime-rs/crates"];
const SOURCE_EXTENSIONS = new Set([
  ".cjs",
  ".js",
  ".jsx",
  ".mjs",
  ".rs",
  ".ts",
  ".tsx",
]);
const IGNORED_DIRECTORIES = new Set([
  ".git",
  "build",
  "coverage",
  "dist",
  "fixtures",
  "node_modules",
  "target",
  "test",
  "tests",
  "__tests__",
]);
const ALLOWED_DECISIONS = new Set([
  "adopted-mechanism",
  "existing-current-owner",
  "excluded",
]);
const ALLOWED_STATUSES = new Set([
  "excluded",
  "open-platform",
  "verified-gate-b",
  "verified-local",
]);
const ALLOWED_EVIDENCE_LEVELS = new Set([
  "codex-desktop-observed",
  "codex-rust",
  "gate-a",
  "gate-b",
  "goose-static",
  "lime-local",
  "platform-packaged",
]);
const REQUIRED_FORBIDDEN_OWNERS = new Set([
  "goose-acp",
  "goose-session-message",
  "goose-recipe-runtime-storage",
  "goose-autonomous-default",
  "second-runtime",
  "second-catalog",
]);

const SOURCE_RULES = [
  {
    id: "goose-owned-symbol",
    pattern:
      /\b(?:Goose(?:Session|Message|Recipe|Runtime|Server|Client)|goose_(?:session|message|recipe|runtime|server|client))\b/giu,
  },
  {
    id: "acp-runtime-owner",
    pattern:
      /\b(?:spawn_acp_sessions|(?:Acp|ACP)(?:Adapter|Connection|Runtime|Session|Transport|Client|Server|Store|Catalog)|acp_(?:adapter|connection|runtime|session|transport|client|server|store|catalog))\b/giu,
  },
  {
    id: "recipe-runtime-owner",
    pattern:
      /\b(?:Recipe(?:Runtime|Engine|Store|Storage|Scheduler|Catalog)|recipe_(?:runtime|engine|store|storage|scheduler|catalog))\b/giu,
  },
  {
    id: "autonomous-default-owner",
    pattern:
      /\b(?:AutonomousDefault|autonomous_default|default_autonomous)\b/giu,
  },
  {
    id: "foreign-protocol-method",
    pattern: /["'`](?:goose|acp|recipe)\/[a-z][a-z0-9_/-]*["'`]/giu,
  },
];

const FORBIDDEN_PATH_PATTERN =
  /(?:^|\/)(?:goose(?:[-_][^/]*)?|acp(?:[-_](?:adapter|runtime|session|transport|server|client|store|catalog))?|recipe[-_](?:runtime|engine|store|storage|scheduler|catalog))(?:\/|$)/iu;
const FORBIDDEN_DEPENDENCY_PATTERN =
  /(?:^|[-_/.@])(?:goose(?:-ai)?|acp|agent-client-protocol|agentclientprotocol)(?:$|[-_/.])/iu;

function normalizePath(filePath) {
  return filePath.replaceAll("\\", "/").replace(/^\.\//u, "");
}

function isTestOnlyPath(filePath) {
  const normalized = normalizePath(filePath);
  const basename = path.posix.basename(normalized);
  const segments = normalized.split("/");
  return (
    segments.some((segment) => IGNORED_DIRECTORIES.has(segment)) ||
    /(?:^|\.)(?:test|spec|fixture)\./u.test(basename) ||
    /_(?:test|tests)\.rs$/u.test(basename)
  );
}

function walkSourceFiles(repoRoot, relativeRoot, output) {
  const absoluteRoot = path.join(repoRoot, relativeRoot);
  if (!fs.existsSync(absoluteRoot)) {
    return;
  }

  for (const entry of fs.readdirSync(absoluteRoot, { withFileTypes: true })) {
    const relativePath = normalizePath(path.join(relativeRoot, entry.name));
    if (entry.isDirectory()) {
      if (!IGNORED_DIRECTORIES.has(entry.name)) {
        walkSourceFiles(repoRoot, relativePath, output);
      }
      continue;
    }
    if (
      entry.isFile() &&
      SOURCE_EXTENSIONS.has(path.extname(entry.name)) &&
      !isTestOnlyPath(relativePath)
    ) {
      output.push({
        path: relativePath,
        content: fs.readFileSync(path.join(repoRoot, relativePath), "utf8"),
      });
    }
  }
}

function lineNumberAt(content, index) {
  return content.slice(0, index).split("\n").length;
}

export function scanSourceRecords(records) {
  const violations = [];
  for (const record of records) {
    const normalizedPath = normalizePath(record.path);
    if (FORBIDDEN_PATH_PATTERN.test(normalizedPath)) {
      violations.push({
        kind: "source-path",
        rule: "parallel-owner-path",
        path: normalizedPath,
      });
    }

    for (const rule of SOURCE_RULES) {
      const pattern = new RegExp(rule.pattern.source, rule.pattern.flags);
      for (const match of record.content.matchAll(pattern)) {
        violations.push({
          kind: "source-symbol",
          rule: rule.id,
          path: normalizedPath,
          line: lineNumberAt(record.content, match.index ?? 0),
          match: match[0],
        });
      }
    }
  }
  return violations;
}

function collectPackageDependencyNames(manifest) {
  const names = new Set();
  for (const section of [
    "dependencies",
    "devDependencies",
    "optionalDependencies",
    "peerDependencies",
  ]) {
    const dependencies = manifest?.[section];
    if (!dependencies || typeof dependencies !== "object") {
      continue;
    }
    for (const name of Object.keys(dependencies)) {
      names.add(name);
    }
  }
  return [...names];
}

export function inspectPackageManifests(records) {
  const violations = [];
  for (const record of records) {
    const manifest = JSON.parse(record.content);
    for (const dependency of collectPackageDependencyNames(manifest)) {
      if (FORBIDDEN_DEPENDENCY_PATTERN.test(dependency)) {
        violations.push({
          kind: "dependency",
          rule: "foreign-runtime-dependency",
          path: normalizePath(record.path),
          match: dependency,
        });
      }
    }
  }
  return violations;
}

export function inspectCargoManifests(records) {
  const violations = [];
  for (const record of records) {
    let section = "";
    for (const [index, line] of record.content.split(/\r?\n/u).entries()) {
      const sectionMatch = line.match(/^\s*\[([^\u005d]+)\]\s*$/u);
      if (sectionMatch) {
        section = sectionMatch[1];
        continue;
      }
      if (!/(?:^|\.)(?:build-|dev-)?dependencies(?:\.|$)/u.test(section)) {
        continue;
      }
      const dependencyMatch = line.match(/^\s*([a-zA-Z0-9_-]+)\s*=/u);
      const packageMatch = line.match(/\bpackage\s*=\s*["']([^"']+)["']/u);
      for (const dependency of [dependencyMatch?.[1], packageMatch?.[1]]) {
        if (dependency && FORBIDDEN_DEPENDENCY_PATTERN.test(dependency)) {
          violations.push({
            kind: "dependency",
            rule: "foreign-runtime-dependency",
            path: normalizePath(record.path),
            line: index + 1,
            match: dependency,
          });
        }
      }
    }
  }
  return violations;
}

function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}

function requireStringArray(errors, value, label, { allowEmpty = false } = {}) {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0)) {
    errors.push(
      `${label} must be ${allowEmpty ? "an array" : "a non-empty array"}`,
    );
    return;
  }
  value.forEach((item, index) => {
    if (!isNonEmptyString(item)) {
      errors.push(`${label}[${index}] must be a non-empty string`);
    }
  });
}

export function validateEvidenceIndex(
  index,
  { pathExists = (relativePath) => fs.existsSync(relativePath) } = {},
) {
  const errors = [];
  if (index?.schemaVersion !== 1) {
    errors.push("schemaVersion must be 1");
  }
  if (index?.productTarget !== "codex-desktop") {
    errors.push("productTarget must be codex-desktop");
  }
  if (index?.productChain !== PRODUCT_CHAIN) {
    errors.push("productChain must remain on the canonical desktop chain");
  }
  if (!isNonEmptyString(index?.planPath) || !pathExists(index.planPath)) {
    errors.push("planPath must point to an existing execution plan");
  }

  const reference = index?.reference;
  if (reference?.repository !== "https://github.com/aaif-goose/goose") {
    errors.push("reference.repository must identify the reviewed Goose source");
  }
  if (!/^[a-f0-9]{40}$/u.test(reference?.commit ?? "")) {
    errors.push("reference.commit must be a full Git commit SHA");
  }
  if (reference?.license !== "Apache-2.0") {
    errors.push("reference.license must be Apache-2.0");
  }
  if (reference?.role !== "mechanism-reference-only") {
    errors.push("reference.role must be mechanism-reference-only");
  }
  if (reference?.codeCopied !== false) {
    errors.push(
      "reference.codeCopied must remain false until a licensed copy record exists",
    );
  }
  if (reference?.dependencyAdded !== false) {
    errors.push("reference.dependencyAdded must remain false");
  }

  requireStringArray(errors, index?.forbiddenOwners, "forbiddenOwners");
  const forbiddenOwners = new Set(index?.forbiddenOwners ?? []);
  for (const owner of REQUIRED_FORBIDDEN_OWNERS) {
    if (!forbiddenOwners.has(owner)) {
      errors.push(`forbiddenOwners is missing ${owner}`);
    }
  }

  if (!Array.isArray(index?.entries) || index.entries.length === 0) {
    errors.push("entries must be a non-empty array");
    return errors;
  }

  const ids = new Set();
  for (const [entryIndex, entry] of index.entries.entries()) {
    const label = `entries[${entryIndex}]`;
    if (!isNonEmptyString(entry?.id)) {
      errors.push(`${label}.id must be a non-empty string`);
    } else if (ids.has(entry.id)) {
      errors.push(`${label}.id is duplicated: ${entry.id}`);
    } else {
      ids.add(entry.id);
    }
    if (!ALLOWED_DECISIONS.has(entry?.decision)) {
      errors.push(`${label}.decision is invalid`);
    }
    if (!ALLOWED_STATUSES.has(entry?.status)) {
      errors.push(`${label}.status is invalid`);
    }
    requireStringArray(errors, entry?.targetBasis, `${label}.targetBasis`);
    if (
      Array.isArray(entry?.targetBasis) &&
      !entry.targetBasis.some((basis) =>
        ["codex-desktop-observed", "codex-rust"].includes(basis),
      )
    ) {
      errors.push(`${label}.targetBasis cannot rely on Goose alone`);
    }
    for (const level of entry?.evidenceLevels ?? []) {
      if (!ALLOWED_EVIDENCE_LEVELS.has(level)) {
        errors.push(`${label}.evidenceLevels contains unknown level: ${level}`);
      }
    }
    requireStringArray(
      errors,
      entry?.evidenceLevels,
      `${label}.evidenceLevels`,
    );
    requireStringArray(
      errors,
      entry?.implementationPaths,
      `${label}.implementationPaths`,
      { allowEmpty: entry?.decision === "excluded" },
    );
    for (const implementationPath of entry?.implementationPaths ?? []) {
      if (
        isNonEmptyString(implementationPath) &&
        !pathExists(implementationPath)
      ) {
        errors.push(
          `${label}.implementationPaths is missing ${implementationPath}`,
        );
      }
    }

    if (entry?.decision === "excluded") {
      requireStringArray(errors, entry?.guardIds, `${label}.guardIds`);
    } else {
      if (!isNonEmptyString(entry?.limeOwner)) {
        errors.push(`${label}.limeOwner must name the current owner`);
      }
      requireStringArray(
        errors,
        entry?.verificationCommands,
        `${label}.verificationCommands`,
      );
    }
    if (entry?.status === "open-platform") {
      requireStringArray(errors, entry?.openRefs, `${label}.openRefs`);
    }
  }

  return errors;
}

function collectFiles(repoRoot, root, fileName, output) {
  const absoluteRoot = path.join(repoRoot, root);
  if (!fs.existsSync(absoluteRoot)) {
    return;
  }
  for (const entry of fs.readdirSync(absoluteRoot, { withFileTypes: true })) {
    if (IGNORED_DIRECTORIES.has(entry.name)) {
      continue;
    }
    const relativePath = normalizePath(path.join(root, entry.name));
    if (entry.isDirectory()) {
      collectFiles(repoRoot, relativePath, fileName, output);
    } else if (entry.isFile() && entry.name === fileName) {
      output.push({
        path: relativePath,
        content: fs.readFileSync(path.join(repoRoot, relativePath), "utf8"),
      });
    }
  }
}

export function buildDesktopReferenceBoundaryReport(repoRoot = process.cwd()) {
  const sources = [];
  for (const root of SOURCE_ROOTS) {
    walkSourceFiles(repoRoot, root, sources);
  }

  const packageManifests = [
    {
      path: "package.json",
      content: fs.readFileSync(path.join(repoRoot, "package.json"), "utf8"),
    },
  ];
  collectFiles(repoRoot, "packages", "package.json", packageManifests);

  const cargoManifests = [];
  collectFiles(repoRoot, "lime-rs", "Cargo.toml", cargoManifests);

  const evidencePath = path.join(repoRoot, EVIDENCE_INDEX_PATH);
  const evidenceIndex = JSON.parse(fs.readFileSync(evidencePath, "utf8"));
  const evidenceErrors = validateEvidenceIndex(evidenceIndex, {
    pathExists: (relativePath) =>
      fs.existsSync(path.join(repoRoot, relativePath)),
  });
  const violations = [
    ...scanSourceRecords(sources),
    ...inspectPackageManifests(packageManifests),
    ...inspectCargoManifests(cargoManifests),
  ];

  return {
    sourceCount: sources.length,
    packageManifestCount: packageManifests.length,
    cargoManifestCount: cargoManifests.length,
    evidenceEntryCount: evidenceIndex.entries?.length ?? 0,
    evidenceErrors,
    violations,
  };
}

function printReport(report) {
  for (const violation of report.violations) {
    const location = violation.line
      ? `${violation.path}:${violation.line}`
      : violation.path;
    console.error(
      `[desktop-reference-boundary] ${violation.rule} ${location}: ${violation.match ?? "forbidden path"}`,
    );
  }
  for (const error of report.evidenceErrors) {
    console.error(`[desktop-reference-boundary] evidence: ${error}`);
  }

  if (report.violations.length === 0 && report.evidenceErrors.length === 0) {
    console.log(
      `[desktop-reference-boundary] ok sources=${report.sourceCount} packageManifests=${report.packageManifestCount} cargoManifests=${report.cargoManifestCount} evidenceEntries=${report.evidenceEntryCount}`,
    );
  }
}

const isMainModule =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isMainModule) {
  try {
    const report = buildDesktopReferenceBoundaryReport();
    printReport(report);
    if (report.violations.length > 0 || report.evidenceErrors.length > 0) {
      process.exitCode = 1;
    }
  } catch (error) {
    console.error(
      `[desktop-reference-boundary] ${error instanceof Error ? error.message : String(error)}`,
    );
    process.exitCode = 1;
  }
}
