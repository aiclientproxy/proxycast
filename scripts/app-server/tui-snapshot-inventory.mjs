#!/usr/bin/env node

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "../..");
const referenceDir = path.resolve(
  process.env.CODEX_TUI_REFERENCE ||
    "/Users/coso/Documents/dev/rust/codex/codex-rs/tui",
);
const outputPath = path.join(
  rootDir,
  "internal/exec-plans/tui-codex-snapshot-inventory.json",
);

const deadProductMarkers = [
  "external_agent_config_migration",
  "model_migration",
  "onboarding",
  "update_prompt",
  "update_available",
  "update_popup",
  "feedback",
  "rate_limit",
  "signed_out",
  "chatgpt",
  "account_change",
  "marketplace",
  "cyber",
  "luna_reserve",
  "debug_config",
];

const rules = [
  {
    id: "dead-codex-product",
    classification: "dead",
    matches: (relativePath) =>
      !relativePath.includes("hook_blocked_failed_feedback_history") &&
      deadProductMarkers.some((marker) => relativePath.includes(marker)),
    limeOwner: null,
    rationale:
      "Codex account, onboarding, migration, update, marketplace, or product-only behavior is not a Lime TUI contract.",
  },
  {
    id: "direct-terminal-algorithm",
    classification: "direct",
    modules: [
      "diff_render",
      "markdown_render",
      "insert_history",
      "render",
      "terminal_hyperlinks",
      "terminal_palette",
      "table_detect",
      "wrapping",
    ],
    limeOwner: "tui::{diff_render,markdown_render,view,entry}",
    rationale:
      "The snapshot belongs to a pure terminal algorithm that can be adapted without importing Codex runtime state.",
  },
  {
    id: "contract-runtime-lifecycle",
    classification: "contract",
    modules: [
      "app",
      "app_backtrack",
      "cwd_prompt",
      "multi_agents",
      "resume_picker",
      "unarchive_prompt",
    ],
    limeOwner: "App Server JSON-RPC -> tui::{app,projection,resume_picker}",
    rationale:
      "The surface crosses runtime or persisted state ownership and must be rebuilt on canonical App Server contracts.",
  },
  {
    id: "defer-non-priority-terminal",
    classification: "defer",
    modules: [
      "custom_terminal",
      "git_action_directives",
      "inline_visualization",
      "keymap_setup",
      "startup_hooks_review",
    ],
    limeOwner: "tui",
    rationale:
      "The behavior may be useful later but is outside the current Codex Desktop parity priority.",
  },
  {
    id: "merge-interaction-surface",
    classification: "merge",
    limeOwner: "tui::{app,composer,entry,view,model_picker,resume_picker}",
    rationale:
      "The interaction is relevant, but must consume Lime canonical projection and product copy.",
  },
];

async function main() {
  const paths = (await walk(referenceDir))
    .filter((candidate) => candidate.endsWith(".snap"))
    .sort();
  if (paths.length === 0) {
    throw new Error(`no Codex TUI snapshots found under ${referenceDir}`);
  }

  const entries = await Promise.all(
    paths.map(async (snapshotPath) => {
      const relativePath = path
        .relative(referenceDir, snapshotPath)
        .split(path.sep)
        .join("/");
      const content = await readFile(snapshotPath);
      const module = snapshotModule(relativePath);
      const rule = rules.find(
        (candidate) =>
          candidate.matches?.(relativePath) ||
          candidate.modules?.includes(module) ||
          candidate.id === "merge-interaction-surface",
      );
      if (!rule) {
        throw new Error(`unclassified snapshot: ${relativePath}`);
      }
      return {
        path: relativePath,
        module,
        sha256: createHash("sha256").update(content).digest("hex"),
        classification: rule.classification,
        rule: rule.id,
      };
    }),
  );
  const counts = Object.fromEntries(
    ["direct", "merge", "contract", "defer", "dead"].map((classification) => [
      classification,
      entries.filter((entry) => entry.classification === classification).length,
    ]),
  );
  const inventory = {
    schemaVersion: 1,
    source: "codex-rs/tui",
    sourceCommit: execFileSync(
      "git",
      ["-C", referenceDir, "rev-parse", "HEAD"],
      { encoding: "utf8" },
    ).trim(),
    snapshotCount: entries.length,
    sourcePathSetSha256: createHash("sha256")
      .update(
        entries
          .map((entry) => entry.path.replace(/^src\//u, ""))
          .join("\n") + "\n",
      )
      .digest("hex"),
    counts,
    rules: rules.map(({ matches: _matches, ...rule }) => rule),
    entries,
  };
  await writeFile(outputPath, `${JSON.stringify(inventory, null, 2)}\n`, "utf8");
  console.log(
    `[inventory:tui-codex] wrote ${entries.length} snapshots to ${path.relative(rootDir, outputPath)} ${JSON.stringify(counts)}`,
  );
}

function snapshotModule(relativePath) {
  const basename = path.basename(relativePath, ".snap");
  const [, module] = basename.split("__");
  if (!module) {
    throw new Error(`snapshot does not expose an insta module: ${relativePath}`);
  }
  return module;
}

async function walk(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const candidate = path.join(directory, entry.name);
      return entry.isDirectory() ? walk(candidate) : [candidate];
    }),
  );
  return nested.flat();
}

main().catch((error) => {
  console.error(
    `[inventory:tui-codex] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
