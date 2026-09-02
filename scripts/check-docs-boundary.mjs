#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { withNativeSystemPath } from "./lib/native-executable-env.mjs";

const repoRoot = process.cwd();
const docsRoot = path.join(repoRoot, "docs");

const internalOnlyDirectories = [
  "aiprompts",
  "bussniss",
  "design",
  "develop",
  "exec-plans",
  "gongzonghao",
  "iteration-notes",
  "knowledge",
  "oem",
  "prd",
  "research",
  "roadmap",
  "tech",
  "test",
  "testing",
  "tests",
];

const requiredFiles = ["docs/README.md", "internal/README.md"];
const requiredIgnoreRules = [
  "docs/.data/",
  "docs/.nuxt/",
  "docs/.output/",
  "internal/prd/**",
  "internal/exec-plans/",
  "internal/roadmap/**",
  "internal/gongzonghao/",
  "internal/bussniss/",
  "internal/oem/",
  "internal/tech/",
  "internal/knowledge",
  "internal/research/**",
];
const skippedDirectories = new Set([
  ".git",
  ".lime",
  ".tmp",
  ".tmp-smoke",
  "node_modules",
  "target",
  "dist",
  "lime-rs/target",
]);
const skippedExtensions = new Set([
  ".gif",
  ".ico",
  ".jpeg",
  ".jpg",
  ".pdf",
  ".png",
  ".webp",
  ".zip",
]);

const pluginV3CurrentDocuments = [
  "internal/aiprompts/README.md",
  "internal/roadmap/Writing/v2/README.md",
  "internal/roadmap/Writing/v2/product-requirements.md",
  "internal/roadmap/Writing/v2/content-factory-plugin-reframe.md",
  "internal/roadmap/plugin/v3/00-current-baseline.md",
  "internal/roadmap/plugin/v3/01-target-contract.md",
  "internal/roadmap/plugin/v3/03-execution-plan.md",
  "internal/roadmap/plugin/v3/04-verification.md",
  "internal/roadmap/plugin/v3/05-codex-parity-matrix.md",
];

const retiredPluginContractPatterns = [
  /lime-plugin-package-v1\.md/,
  /lime\.plugin\.package\.v1/,
  /app\.runtime\.yaml/,
  /app\.workbench\.yaml/,
  /contributions\.runtime/,
  /contributions\.workbench/,
  /internal\/tech\/plugin(?:\/|[)`'"\s]|$)/,
];

function toRelativePath(absolutePath) {
  return path.relative(repoRoot, absolutePath).split(path.sep).join("/");
}

function fileExists(relativePath) {
  return fs.existsSync(path.join(repoRoot, relativePath));
}

function isCurrentFile(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  return fs.existsSync(absolutePath) && fs.statSync(absolutePath).isFile();
}

function readText(relativePath) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

function isSkippedDirectory(relativeDirectory) {
  return [...skippedDirectories].some(
    (skippedDirectory) =>
      relativeDirectory === skippedDirectory ||
      relativeDirectory.startsWith(`${skippedDirectory}/`),
  );
}

function listRepoFiles() {
  try {
    const output = execFileSync(
      "git",
      ["ls-files", "-co", "--exclude-standard", "-z"],
      {
        cwd: repoRoot,
        encoding: "utf8",
        env: withNativeSystemPath(process.env),
      },
    );
    return output.split("\0").filter(isCurrentFile);
  } catch {
    return listFilesRecursively(repoRoot).map(toRelativePath);
  }
}

function listTrackedIgnoredInternalFiles() {
  try {
    const output = execFileSync(
      "git",
      ["ls-files", "-ci", "--exclude-standard", "-z", "internal"],
      {
        cwd: repoRoot,
        encoding: "utf8",
        env: withNativeSystemPath(process.env),
      },
    );
    // `git ls-files -ci` 会包含已从工作树删除但尚未提交的路径。文档边界
    // 只约束当前磁盘上的文件，不能要求恢复已删除的内部文档才能通过检查。
    return output.split("\0").filter(isCurrentFile);
  } catch {
    return [];
  }
}

function listFilesRecursively(directoryPath) {
  const relativeDirectory = toRelativePath(directoryPath);
  if (isSkippedDirectory(relativeDirectory)) {
    return [];
  }

  const entries = fs.readdirSync(directoryPath, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const absolutePath = path.join(directoryPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...listFilesRecursively(absolutePath));
      continue;
    }

    if (entry.isFile()) {
      files.push(absolutePath);
    }
  }

  return files;
}

function shouldScanFile(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  if (!isCurrentFile(relativePath)) {
    return false;
  }

  if (
    relativePath === "node_modules" ||
    relativePath.startsWith("node_modules/")
  ) {
    return false;
  }

  if (relativePath.includes("/node_modules/")) {
    return false;
  }

  if (relativePath.startsWith("lime-rs/target")) {
    return false;
  }

  const extension = path.extname(relativePath).toLowerCase();
  if (skippedExtensions.has(extension)) {
    return false;
  }

  const stat = fs.statSync(absolutePath);
  return stat.size <= 2 * 1024 * 1024;
}

function checkPluginV3DocumentationBoundary(failures) {
  for (const relativePath of pluginV3CurrentDocuments) {
    if (!isCurrentFile(relativePath)) {
      failures.push(`Plugin v3 现役文档缺失: ${relativePath}`);
      continue;
    }

    const source = readText(relativePath);
    for (const pattern of retiredPluginContractPatterns) {
      if (pattern.test(source)) {
        failures.push(
          `Plugin v3 现役文档回流旧合同 ${pattern}: ${relativePath}`,
        );
      }
    }
  }
}

function main() {
  const failures = [];

  checkPluginV3DocumentationBoundary(failures);

  for (const relativePath of requiredFiles) {
    if (!fileExists(relativePath)) {
      failures.push(`缺少边界入口文件: ${relativePath}`);
    }
  }

  if (!fs.existsSync(docsRoot)) {
    failures.push("缺少 docs/ 文档站目录");
  } else {
    const entries = fs.readdirSync(docsRoot, { withFileTypes: true });
    const forbiddenDocsEntries = entries
      .map((entry) => entry.name)
      .filter((entryName) => internalOnlyDirectories.includes(entryName));

    for (const entryName of forbiddenDocsEntries) {
      failures.push(
        `docs/${entryName}/ 属于内部事实源，请迁移到 internal/${entryName}/`,
      );
    }
  }

  if (fileExists(".gitignore")) {
    const gitignore = readText(".gitignore");

    for (const directoryName of internalOnlyDirectories) {
      const escapedDirectoryName = directoryName.replaceAll("-", "\\-");
      const docsIgnoreRulePattern = new RegExp(
        `(^|\\n)docs/${escapedDirectoryName}(/|\\*|\\n|$)`,
      );
      if (docsIgnoreRulePattern.test(gitignore)) {
        failures.push(
          `.gitignore 仍在忽略旧 docs/${directoryName} 路径，请改为 internal/${directoryName}`,
        );
      }
    }

    for (const ignoreRule of requiredIgnoreRules) {
      if (!gitignore.includes(ignoreRule)) {
        failures.push(`.gitignore 缺少内部目录忽略规则: ${ignoreRule}`);
      }
    }
  }

  // 只拦截独立的 docs 根路径；例如 `catalog/docs/tests` 不是旧文档根目录。
  const oldDocsReferencePattern = new RegExp(
    `(^|[^A-Za-z0-9_./-])docs/(?:${internalOnlyDirectories.join(
      "|",
    )})(?:/|\\b)`,
  );
  const filesWithOldReferences = [];

  for (const relativePath of listRepoFiles()) {
    if (!shouldScanFile(relativePath)) {
      continue;
    }

    const source = readText(relativePath);
    if (oldDocsReferencePattern.test(source)) {
      filesWithOldReferences.push(relativePath);
    }
  }

  if (filesWithOldReferences.length > 0) {
    failures.push(
      `发现旧内部文档路径引用，请改为 internal/: ${filesWithOldReferences.join(", ")}`,
    );
  }

  const trackedIgnoredInternalFiles = listTrackedIgnoredInternalFiles();
  if (trackedIgnoredInternalFiles.length > 0) {
    failures.push(
      `发现已进入 Git 索引但应被忽略的 internal 文件: ${trackedIgnoredInternalFiles.join(", ")}`,
    );
  }

  if (failures.length > 0) {
    console.error("docs 边界检查失败:");
    for (const failure of failures) {
      console.error(`- ${failure}`);
    }
    process.exit(1);
  }

  console.log("docs 边界检查通过");
}

main();
