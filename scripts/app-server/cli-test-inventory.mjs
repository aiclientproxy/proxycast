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
  process.env.CODEX_CLI_REFERENCE ||
    "/Users/coso/Documents/dev/rust/codex/codex-rs/cli",
);
const outputPath = path.join(
  rootDir,
  "internal/exec-plans/cli-codex-test-inventory.json",
);

const rules = [
  {
    id: "cloud-remote-transport",
    classification: "cloud-deferred",
    status: "deferred",
    limeOwner: "app-server-client authenticated remote transport",
    rationale:
      "Remote-control, Cloud, and exec-server behavior remains deferred beyond the authenticated transport foundation and must not create a second runtime.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)(?:cloud_config|remote_control_cmd|exec_server(?:_telemetry)?(?:_tests)?)(?:\.rs|\/)/u.test(
        relativePath,
      ) ||
      /(?:^|\/)tests\/(?:cloud_auth|cloud_config|exec_server|sandbox_cloud_config)\.rs$/u.test(
        relativePath,
      ) ||
      /(?:^|_)(?:remote|exec_server|cloud_managed_permission_profiles?)(?:_|$)/u.test(testName),
  },
  {
    id: "retired-doctor-surface",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "The retired doctor command and its local installation probes are dead/deleted/forbidden-to-restore; reusable diagnostics must live behind a consumed current App Server owner.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)src\/doctor(?:\.rs|\/)/u.test(relativePath) ||
      /(?:^|\/)tests\/doctor[^/]*\.rs$/u.test(relativePath),
  },
  {
    id: "codex-local-config-and-update-only",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "Codex profile-v2, strict-config, updater, managed requirements, and removed feature aliases do not map to Lime's App Server-owned configuration contract.",
    matches: ({ path: relativePath, testName }) =>
      /(?:profile_v2|profile_loader|strict_config|update_command|update_parses|cloud_managed_feature|legacy_(?:configs|linux_sandbox|sandbox_mode)|removed_(?:image_detail|enable_fanout|item_ids))/u.test(
        testName,
      ) || /(?:^|\/)src\/doctor\/updates\.rs$/u.test(relativePath),
  },
  {
    id: "codex-product-only",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "Codex account, marketplace, updater, desktop launcher, state migration, or internal support-client behavior is not a Lime CLI contract.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)(?:login|marketplace_cmd|app_cmd|state_db_recovery|migrate_rollouts)\.rs$/u.test(
        relativePath,
      ) ||
      /(?:^|\/)desktop_app\//u.test(relativePath) ||
      /(?:^|\/)src\/bin\/logs_client\.rs$/u.test(relativePath) ||
      /(?:^|\/)tests\/(?:login|marketplace_add|marketplace_remove|marketplace_upgrade|update)\.rs$/u.test(
        relativePath,
      ) ||
      /(?:^|_)(?:marketplace|updater|chatgpt)(?:_|$)/u.test(testName),
  },
  {
    id: "current-sandbox-parser",
    classification: "direct",
    status: "covered",
    limeOwner:
      "cli::MultitoolCli -> cli::{SeatbeltCommand, LandlockCommand, WindowsCommand}",
    rationale:
      "Lime directly copies the applicable Codex sandbox parser names, aliases, help visibility, profile parsing, and option dependency tests.",
    matches: ({ path: relativePath, testName }) =>
      relativePath === "src/main.rs" &&
      /^(?:sandbox_parses_permission_profile|sandbox_parses_legacy_permissions_profile_alias|sandbox_help_only_shows_singular_permission_profile|sandbox_parses_permissions_profile_short_alias|sandbox_parses_config_profile|sandbox_rejects_explicit_profile_controls_without_profile)$/u.test(
        testName,
      ),
  },
  {
    id: "current-sandbox-command-exec",
    classification: "contract",
    status: "covered",
    limeOwner:
      "cli::debug_sandbox -> app-server-client -> App Server command/exec -> tool-runtime sandbox",
    rationale:
      "Lime preserves the Codex debug sandbox entry points while owner-level tests and CLI Gate B cover permission-profile lowering, cwd, output, exit status, and read-only fail-closed behavior.",
    matches: ({ path: relativePath, testName }) =>
      relativePath === "src/debug_sandbox.rs" &&
      /^(?:debug_sandbox_honors_active_permission_profiles|debug_sandbox_honors_explicit_builtin_permission_profile|debug_sandbox_honors_explicit_named_permission_profile|debug_sandbox_uses_explicit_cwd)$/u.test(
        testName,
      ),
  },
  {
    id: "current-sandbox-setup",
    classification: "contract",
    status: "covered",
    limeOwner:
      "cli::sandbox_setup -> app-server-client -> App Server windowsSandbox/setupStart -> tool-runtime Windows helper",
    rationale:
      "Lime copies the Codex sandbox_setup module, type, function, and parser test names while delegating actual elevated provisioning to the existing typed App Server owner.",
    matches: ({ path: relativePath }) =>
      relativePath === "src/sandbox_setup.rs",
  },
  {
    id: "current-execpolicy-check",
    classification: "contract",
    status: "covered",
    limeOwner: "cli::ExecpolicyCommand -> execpolicy::ExecPolicyCheckCommand",
    rationale:
      "Lime exposes the Codex-shaped execpolicy check command with prefix-rule matching, strictest decision lowering, optional justification, and JSON/pretty output.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)tests\/execpolicy\.rs$/u.test(relativePath),
  },
  {
    id: "codex-platform-sandbox-internals",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "Codex macOS PID tracking is coupled to its seatbelt denial logger and process-kill implementation; Lime keeps process lifecycle in App Server/tool-runtime and does not expose this private CLI module.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)src\/debug_sandbox\/pid_tracker\.rs$/u.test(relativePath),
  },
  {
    id: "codex-code-mode-host-url-contract",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "Codex app-server --code-mode-host accepts a hosted HTTP endpoint; Lime's current Code Mode owner is a packaged local code-mode-host process and has no remote URL contract.",
    matches: ({ path: relativePath, testName }) =>
      relativePath === "src/main.rs" &&
      testName === "app_server_rejects_invalid_code_mode_host_urls",
  },
  {
    id: "codex-network-proxy-sandbox",
    classification: "product-specific",
    status: "excluded",
    limeOwner: null,
    rationale:
      "Codex bubblewrap network-proxy loopback tests depend on its feature flag and proxy process; Lime has no equivalent local proxy owner and continues to fail closed instead of emulating it in CLI.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)tests\/sandbox_network_proxy\.rs$/u.test(relativePath),
  },
  {
    id: "current-sandbox-and-execpolicy",
    classification: "missing",
    status: "pending",
    limeOwner:
      "cli -> app-server-client -> App Server command/exec -> tool-runtime sandbox/execpolicy",
    rationale:
      "The remaining Codex sandbox-state replay, managed network, and named-profile cases still need current-owner implementations and integration tests.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)src\/debug_sandbox(?:\.rs|\/)/u.test(relativePath) ||
      /(?:^|\/)tests\/sandbox_network_proxy\.rs$/u.test(relativePath) ||
      /(?:^|_)sandbox_(?:parses|help|rejects)(?:_|$)/u.test(testName),
  },
  {
    id: "current-debug-control-plane",
    classification: "contract",
    status: "covered",
    limeOwner: "cli::debug_cmd -> App Server model/list + memory/reset",
    rationale:
      "Rust parsing and the real CLI surface Gate B cover debug models and memory reset through typed App Server methods.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/debug_(?:models|clear_memories)\.rs$/u.test(
        relativePath,
      ) || /(?:^|_)debug_models_parses_bundled_flag$/u.test(testName),
  },
  {
    id: "current-feature-control-plane",
    classification: "contract",
    status: "covered",
    limeOwner: "cli::features_cmd -> App Server experimentalFeature/*",
    rationale:
      "Feature list, sorted rendering, enable/disable mutation, unknown-key rejection, and under-development warning behavior use the current typed App Server owner and CLI Gate B.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|_)(?:features_(?:enable|disable)|feature_toggles_(?:known|unknown)|features_list|features_enable_under_development)(?:_|$)/u.test(
        testName,
      ) || /(?:^|\/)tests\/features\.rs$/u.test(relativePath),
  },
  {
    id: "current-plugin-control-plane",
    classification: "contract",
    status: "partial",
    limeOwner: "cli::plugin_cmd -> App Server plugin/*",
    rationale:
      "The real CLI surface Gate B covers add/list/read/search/enable/disable/remove; Codex marketplace-cache-specific edge cases do not share Lime's implementation.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/plugin_cli\.rs$/u.test(relativePath) ||
      /(?:^|_)plugin_(?:add|list|remove)(?:_|$)/u.test(testName),
  },
  {
    id: "current-mcp-mutation-control-plane",
    classification: "contract",
    status: "partial",
    limeOwner: "cli::mcp_cmd -> App Server mcpServer/*",
    rationale:
      "The real CLI surface Gate B covers list/add/get/remove/start/stop and typed parsing; OAuth registration/logout gaps remain explicit and fail closed.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)tests\/mcp_add_remove\.rs$/u.test(relativePath),
  },
  {
    id: "current-queue-control-plane",
    classification: "contract",
    status: "partial",
    limeOwner: "cli::queue_cmd -> App Server thread/queue/*",
    rationale:
      "Queue add/list, empty input rejection, unavailable Thread failure, and canonical queued submission behavior are covered locally; authenticated remote queue evidence remains deferred with Cloud.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)tests\/queue\.rs$/u.test(relativePath),
  },
  {
    id: "current-app-server-entrypoint",
    classification: "contract",
    status: "partial",
    limeOwner: "cli::app_server_cmd -> sibling app-server",
    rationale:
      "The CLI preserves and forwards the sibling App Server argument surface; App Server transport parsing remains owned and tested by the App Server crate.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/app_server\.rs$/u.test(relativePath) ||
      /(?:^|_)app_server_(?:analytics|grpc|listen|stdio|proxy|daemon|capability|signed|rejects_removed)(?:_|$)/u.test(
        testName,
      ),
  },
  {
    id: "current-cli-parser-contract",
    classification: "direct",
    status: "covered",
    limeOwner: "cli",
    rationale:
      "The Codex-shaped parser behavior is directly covered by Lime's canonical clap command tree.",
    matches: ({ testName }) =>
      /(?:^|_)(?:responses_subcommand_is_not_registered|plugin_(?:add|list|remove)_parses_under_plugin|features_(?:enable|disable)_parses_feature_name)(?:_|$)/u.test(
        testName,
      ),
  },
  {
    id: "current-permission-options",
    classification: "contract",
    status: "partial",
    limeOwner:
      "cli TuiCli/ExecCli -> App Server permissionProfile/list + turn/start",
    rationale:
      "Lime supports explicit permission profiles and current runtime lowering, while Codex-only approve-for-me aliases and merge precedence are not copied verbatim.",
    matches: ({ testName }) =>
      /(?:approve_for_me|not_so_yolo|dangerous_bypass|approval_policy|exec_sandbox|interactive_permissions)/u.test(
        testName,
      ),
  },
  {
    id: "current-thread-lifecycle",
    classification: "contract",
    status: "partial",
    limeOwner: "App Server JSON-RPC thread/* -> cli/tui",
    rationale:
      "Lime exposes canonical resume, archive, delete, unarchive, and fork behavior through App Server methods.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/delete\.rs$/u.test(relativePath) ||
      /(?:^|_)(?:resume|archive|unarchive|delete|fork)(?:_|$)/u.test(testName),
  },
  {
    id: "current-cli-exec-and-tui",
    classification: "contract",
    status: "covered",
    limeOwner: "cli -> app-server-client -> App Server JSON-RPC -> RuntimeCore",
    rationale:
      "The behavior is covered by the current interactive TUI or non-interactive exec product chain, with real PTY and CLI Gate B evidence proving the App Server JSON-RPC path.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/app_server\.rs$/u.test(relativePath) &&
      testName === "app_server_emits_json_info_events"
        ? false
        : /(?:interactive_tui|format_exit_messages|completion|exec_(?:alias|resume|output)|prompt)/u.test(
            testName,
          ),
  },
  {
    id: "current-mcp-status",
    classification: "contract",
    status: "covered",
    limeOwner: "App Server JSON-RPC mcpServer/list -> cli",
    rationale:
      "Lime mcp list/get and lifecycle Gate B consume the typed current App Server mcpServer/list catalog; runtime status remains a separate GUI control-plane catalog.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)tests\/mcp_list\.rs$/u.test(relativePath) ||
      /(?:^|_)mcp_list(?:_|$)/u.test(testName),
  },
  {
    id: "current-cli-pure-contract",
    classification: "direct",
    status: "covered",
    limeOwner: "cli",
    rationale:
      "The pure command parsing, output, or process-exit behavior has a direct Lime CLI equivalent.",
    matches: ({ path: relativePath, testName }) =>
      /(?:^|\/)exit_status\.rs$/u.test(relativePath) ||
      /(?:^|_)(?:completion|exit_status)(?:_|$)/u.test(testName),
  },
  {
    id: "portable-wsl-paths",
    classification: "direct",
    status: "covered",
    limeOwner: "cli::wsl_paths",
    rationale:
      "The Codex WSL path algorithms and test names are implemented by the canonical cli::wsl_paths owner.",
    matches: ({ path: relativePath }) =>
      /(?:^|\/)wsl_paths\.rs$/u.test(relativePath),
  },
  {
    id: "missing-current-owner",
    classification: "missing",
    status: "pending",
    limeOwner:
      "cli -> App Server JSON-RPC; tool-runtime for sandbox/execpolicy; diagnostics owner for doctor",
    rationale:
      "The behavior is relevant to Lime but is not yet covered by an equivalent CLI command and test at the current owner boundary.",
    matches: () => true,
  },
];

async function main() {
  const sourcePaths = (await walk(referenceDir))
    .filter((candidate) => candidate.endsWith(".rs"))
    .sort();
  const entries = [];
  const sourceFiles = [];

  for (const sourcePath of sourcePaths) {
    const relativePath = path
      .relative(referenceDir, sourcePath)
      .split(path.sep)
      .join("/");
    const source = await readFile(sourcePath, "utf8");
    const sourceFileSha256 = sha256(source);
    const tests = extractTests(source, relativePath);
    if (tests.length === 0) {
      continue;
    }
    sourceFiles.push({
      path: relativePath,
      sha256: sourceFileSha256,
      testCount: tests.length,
    });
    for (const test of tests) {
      const rule = rules.find((candidate) => candidate.matches(test));
      if (!rule) {
        throw new Error(
          `unclassified CLI test: ${relativePath}::${test.testName}`,
        );
      }
      entries.push({
        ...test,
        sourceFileSha256,
        classification: rule.classification,
        status: rule.status,
        rule: rule.id,
        limeOwner: rule.limeOwner,
        rationale: rule.rationale,
      });
    }
  }

  entries.sort((left, right) =>
    `${left.path}:${String(left.sourceLine).padStart(8, "0")}`.localeCompare(
      `${right.path}:${String(right.sourceLine).padStart(8, "0")}`,
    ),
  );
  const classifications = [
    "direct",
    "contract",
    "product-specific",
    "cloud-deferred",
    "missing",
  ];
  const statuses = ["covered", "partial", "pending", "deferred", "excluded"];
  const inventory = {
    schemaVersion: 1,
    source: "codex-rs/cli",
    sourceCommit: execFileSync(
      "git",
      ["-C", referenceDir, "rev-parse", "HEAD"],
      { encoding: "utf8" },
    ).trim(),
    sourceFileCount: sourceFiles.length,
    testCount: entries.length,
    sourcePathSetSha256: sha256(
      `${entries.map((entry) => `${entry.path}::${entry.testName}`).join("\n")}\n`,
    ),
    sourceTreeSha256: sha256(
      `${sourceFiles.map((file) => `${file.path}\0${file.sha256}`).join("\n")}\n`,
    ),
    counts: Object.fromEntries(
      classifications.map((classification) => [
        classification,
        entries.filter((entry) => entry.classification === classification)
          .length,
      ]),
    ),
    statusCounts: Object.fromEntries(
      statuses.map((status) => [
        status,
        entries.filter((entry) => entry.status === status).length,
      ]),
    ),
    rules: rules.map(({ matches: _matches, ...rule }) => rule),
    sourceFiles,
    entries,
  };
  await writeFile(
    outputPath,
    `${JSON.stringify(inventory, null, 2)}\n`,
    "utf8",
  );
  console.log(
    `[inventory:cli-codex] wrote ${entries.length} tests from ${sourceFiles.length} files to ${path.relative(rootDir, outputPath)} ${JSON.stringify(inventory.counts)}`,
  );
}

function extractTests(source, relativePath) {
  const lines = source.split(/\r?\n/u);
  const tests = [];
  for (let index = 0; index < lines.length; index += 1) {
    const attribute = lines[index].match(
      /^\s*#\[((?:tokio::)?test)(?:\([^\]]*\))?\]\s*$/u,
    );
    if (!attribute) {
      continue;
    }
    let declarationIndex = index + 1;
    while (declarationIndex < lines.length) {
      const declaration = lines[declarationIndex].match(
        /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z][A-Za-z0-9_]*)\s*(?:\(|<)/u,
      );
      if (declaration) {
        tests.push({
          path: relativePath,
          testName: declaration[1],
          sourceLine: index + 1,
          kind: attribute[1],
          suite: relativePath.startsWith("tests/") ? "integration" : "unit",
        });
        break;
      }
      if (
        declarationIndex - index > 12 ||
        /^\s*#\[((?:tokio::)?test)(?:\([^\]]*\))?\]\s*$/u.test(
          lines[declarationIndex],
        )
      ) {
        throw new Error(
          `test attribute without function declaration: ${relativePath}:${index + 1}`,
        );
      }
      declarationIndex += 1;
    }
  }
  return tests;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
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
    `[inventory:cli-codex] failed: ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
});
