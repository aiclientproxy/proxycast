#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "../..");
const codexRoot = path.resolve(
  process.env.CODEX_TUI_REFERENCE || "/Users/coso/Documents/dev/rust/codex",
);
const codexRootDir = path.join(codexRoot, "codex-rs/tui/src");
const limeRootDir = path.join(rootDir, "lime-rs/crates/tui/src");
const outputPath = path.join(
  rootDir,
  "internal/exec-plans/tui-structure-inventory.json",
);

async function main() {
  const codex = await inspectTree(codexRootDir);
  const lime = await inspectTree(limeRootDir);
  const inventory = {
    schemaVersion: 1,
    source: {
      codexRoot,
      codexTuiCommit: await gitHead(path.join(codexRoot, "codex-rs/tui")),
    },
    trees: { "codex-rs/tui/src": codex, "lime-rs/crates/tui/src": lime },
    comparisons: {
      filesMissingInLime: difference(codex.files, lime.files),
      filesOnlyInLime: difference(lime.files, codex.files),
      symbolNamesMissingInLime: difference(
        codex.symbols.map((symbol) => symbol.name),
        lime.symbols.map((symbol) => symbol.name),
      ),
      symbolNamesOnlyInLime: difference(
        lime.symbols.map((symbol) => symbol.name),
        codex.symbols.map((symbol) => symbol.name),
      ),
    },
    rules: [
      "Codex TUI directory, module, type and function names are the baseline.",
      "Runtime lifecycle and persisted state remain App Server owned.",
      "Product-only account, onboarding, pets, updater and hosted Cloud surfaces stay excluded or deferred.",
    ],
  };
  await writeFile(outputPath, `${JSON.stringify(inventory, null, 2)}\n`, "utf8");
  console.log(
    `[inventory:tui-structure] wrote ${codex.files.length + lime.files.length} files to ${path.relative(rootDir, outputPath)}`,
  );
}

async function inspectTree(directory) {
  const files = (await walk(directory))
    .filter((file) => file.endsWith(".rs"))
    .map((file) => path.relative(directory, file).split(path.sep).join("/"))
    .sort();
  const symbols = [];
  const contentHashes = [];
  for (const relativePath of files) {
    const source = await readFile(path.join(directory, relativePath), "utf8");
    contentHashes.push(`${relativePath}\0${sha256(source)}`);
    symbols.push(...extractSymbols(source, relativePath));
  }
  return {
    root: directory,
    fileCount: files.length,
    files,
    symbolCount: symbols.length,
    symbols,
    treeSha256: sha256(`${contentHashes.join("\n")}\n`),
  };
}

function extractSymbols(source, relativePath) {
  const patterns = [
    /^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
    /^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type)\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
  ];
  return patterns
    .flatMap((pattern) =>
      [...source.matchAll(pattern)].map((match) => ({
        path: relativePath,
        name: match[1],
        kind: pattern.source.includes("fn") ? "function" : "type",
      })),
    )
    .sort((left, right) =>
      `${left.path}:${left.name}`.localeCompare(`${right.path}:${right.name}`),
    );
}

function difference(left, right) {
  const rightSet = new Set(right);
  return [...new Set(left)].filter((entry) => !rightSet.has(entry)).sort();
}

async function walk(directory) {
  const entries = (await readdir(directory, { withFileTypes: true })).filter(
    (entry) => entry.name !== "target" && entry.name !== "node_modules",
  );
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const candidate = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(candidate) : [candidate];
    }),
  );
  return nested.flat();
}

async function gitHead(directory) {
  const { execFileSync } = await import("node:child_process");
  return execFileSync("git", ["-C", directory, "rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

main().catch((error) => {
  console.error(
    `[inventory:tui-structure] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
