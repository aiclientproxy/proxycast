#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, "../..");
const codexRoot = path.resolve(
  process.env.CODEX_CLI_REFERENCE || "/Users/coso/Documents/dev/rust/codex",
);
const limeRustRoot = path.join(rootDir, "lime-rs/crates/cli");
const limeExecpolicyRoot = path.join(rootDir, "lime-rs/crates/execpolicy");
const limeNpmRoot = path.join(rootDir, "packages/cli");
const outputPath = path.join(
  rootDir,
  "internal/exec-plans/cli-structure-inventory.json",
);

async function main() {
  const trees = {
    "codex-rs/cli": await inspectTree(
      path.join(codexRoot, "codex-rs/cli"),
      "rust",
    ),
    "codex-rs/execpolicy": await inspectTree(
      path.join(codexRoot, "codex-rs/execpolicy"),
      "rust",
    ),
    "codex-cli": await inspectTree(path.join(codexRoot, "codex-cli"), "node"),
    "lime-rs/crates/cli": await inspectTree(limeRustRoot, "rust"),
    "lime-rs/crates/execpolicy": await inspectTree(limeExecpolicyRoot, "rust"),
    "packages/cli": await inspectTree(limeNpmRoot, "node"),
  };
  const inventory = {
    schemaVersion: 1,
    source: {
      codexRoot,
      codexRsCliCommit: await gitHead(path.join(codexRoot, "codex-rs/cli")),
    },
    trees,
    comparisons: {
      rustFilesMissingInLime: difference(
        trees["codex-rs/cli"].files,
        trees["lime-rs/crates/cli"].files,
      ),
      rustFilesOnlyInLime: difference(
        trees["lime-rs/crates/cli"].files,
        trees["codex-rs/cli"].files,
      ),
      execpolicyFilesMissingInLime: difference(
        trees["codex-rs/execpolicy"].files,
        trees["lime-rs/crates/execpolicy"].files,
      ),
      execpolicyFilesOnlyInLime: difference(
        trees["lime-rs/crates/execpolicy"].files,
        trees["codex-rs/execpolicy"].files,
      ),
      npmFilesMissingInLime: difference(
        trees["codex-cli"].files,
        trees["packages/cli"].files,
      ),
      npmFilesOnlyInLime: difference(
        trees["packages/cli"].files,
        trees["codex-cli"].files,
      ),
      rustSymbolNamesMissingInLime: difference(
        trees["codex-rs/cli"].symbols.map((symbol) => symbol.name),
        trees["lime-rs/crates/cli"].symbols.map((symbol) => symbol.name),
      ),
      rustSymbolNamesOnlyInLime: difference(
        trees["lime-rs/crates/cli"].symbols.map((symbol) => symbol.name),
        trees["codex-rs/cli"].symbols.map((symbol) => symbol.name),
      ),
      execpolicySymbolNamesMissingInLime: difference(
        trees["codex-rs/execpolicy"].symbols.map((symbol) => symbol.name),
        trees["lime-rs/crates/execpolicy"].symbols.map((symbol) => symbol.name),
      ),
      execpolicySymbolNamesOnlyInLime: difference(
        trees["lime-rs/crates/execpolicy"].symbols.map((symbol) => symbol.name),
        trees["codex-rs/execpolicy"].symbols.map((symbol) => symbol.name),
      ),
      npmSymbolNamesMissingInLime: difference(
        trees["codex-cli"].symbols.map((symbol) => symbol.name),
        trees["packages/cli"].symbols.map((symbol) => symbol.name),
      ),
      npmSymbolNamesOnlyInLime: difference(
        trees["packages/cli"].symbols.map((symbol) => symbol.name),
        trees["codex-cli"].symbols.map((symbol) => symbol.name),
      ),
    },
    rules: [
      "Codex directory, module, type and function names are the baseline.",
      "Product-only account, marketplace, updater, desktop and Cloud runtime code stays excluded or deferred with an owner.",
      "Lime CLI and npm launcher must retain the current App Server JSON-RPC chain; no direct runtime or config-file bypass is allowed.",
      "Cloud remains an authenticated app-server-client transport foundation only; production Cloud behavior stays deferred.",
    ],
  };
  await writeFile(
    outputPath,
    `${JSON.stringify(inventory, null, 2)}\n`,
    "utf8",
  );
  console.log(
    `[inventory:cli-structure] wrote ${Object.keys(trees).length} trees to ${path.relative(rootDir, outputPath)}`,
  );
}

async function inspectTree(directory, language) {
  const files = (await walk(directory))
    .filter((file) => !file.includes(`${path.sep}target${path.sep}`))
    .map((file) => path.relative(directory, file).split(path.sep).join("/"))
    .sort();
  const symbols = [];
  const contentHashes = [];
  for (const relativePath of files) {
    const source = await readFile(path.join(directory, relativePath), "utf8");
    contentHashes.push(`${relativePath}\0${sha256(source)}`);
    symbols.push(...extractSymbols(source, relativePath, language));
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

function extractSymbols(source, relativePath, language) {
  const patterns =
    language === "rust"
      ? [
          /^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
          /^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait|type)\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
        ]
      : [
          /^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
          /^(?:export\s+)?(?:const|class|interface|type)\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
          /^def\s+([A-Za-z][A-Za-z0-9_]*)/gmu,
        ];
  return patterns
    .flatMap((pattern) =>
      [...source.matchAll(pattern)].map((match) => ({
        path: relativePath,
        name: match[1],
        kind:
          language === "rust"
            ? pattern.source.includes("fn")
              ? "function"
              : "type"
            : "symbol",
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
    (entry) =>
      entry.name !== "__pycache__" &&
      !entry.name.endsWith(".pyc") &&
      entry.name !== "node_modules",
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
  return execFileSync("git", ["-C", directory, "rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

main().catch((error) => {
  console.error(
    `[inventory:cli-structure] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
