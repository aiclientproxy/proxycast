import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const inventory = JSON.parse(
  readFileSync(
    path.resolve(
      process.cwd(),
      "internal/exec-plans/cli-structure-inventory.json",
    ),
    "utf8",
  ),
);

describe("Codex CLI structure inventory", () => {
  it("records both upstream directories and both Lime owners", () => {
    expect(inventory.schemaVersion).toBe(1);
    expect(Object.keys(inventory.trees)).toEqual([
      "codex-rs/cli",
      "codex-rs/execpolicy",
      "codex-cli",
      "lime-rs/crates/cli",
      "lime-rs/crates/execpolicy",
      "packages/cli",
    ]);
    for (const tree of Object.values(inventory.trees)) {
      expect(tree.fileCount).toBeGreaterThan(0);
      expect(tree.symbolCount).toBeGreaterThan(0);
      expect(tree.treeSha256).toMatch(/^[a-f0-9]{64}$/u);
    }
  });

  it("locks Codex-shaped current names in the Lime command owner", () => {
    const symbols = new Set(
      inventory.trees["lime-rs/crates/cli"].symbols.map(
        (symbol) => symbol.name,
      ),
    );
    for (const name of [
      "MultitoolCli",
      "Subcommand",
      "TuiCli",
      "ExecCli",
      "ResumeCommand",
      "McpCli",
      "PluginCli",
      "FeaturesCli",
      "QueueCommand",
      "DebugCommand",
      "CompletionCommand",
      "SandboxStateArgs",
      "SeatbeltCommand",
      "LandlockCommand",
      "WindowsCommand",
      "ExecpolicyCommand",
      "ExecpolicySubcommand",
      "SandboxSetupCommand",
      "SandboxSetupLevel",
      "SandboxSetupIdentity",
      "handle_exit_status",
    ]) {
      expect(symbols.has(name), name).toBe(true);
    }
    for (const relativePath of [
      "src/debug_sandbox.rs",
      "src/exit_status.rs",
      "src/lib.rs",
      "src/sandbox_setup.rs",
    ]) {
      expect(inventory.comparisons.rustFilesMissingInLime).not.toContain(
        relativePath,
      );
    }
  });

  it("keeps bidirectional structural differences explicit for follow-up work", () => {
    expect(inventory.comparisons.rustFilesMissingInLime).toContain(
      "src/doctor.rs",
    );
    expect(inventory.comparisons.npmFilesMissingInLime).toContain(
      "scripts/init_firewall.sh",
    );
    expect(
      inventory.comparisons.rustSymbolNamesMissingInLime.length,
    ).toBeGreaterThan(0);
    expect(inventory.comparisons.rustFilesOnlyInLime).not.toContain(
      "src/commands.rs",
    );
    expect(inventory.comparisons.rustFilesOnlyInLime).not.toContain(
      "src/execpolicy.rs",
    );
    expect(inventory.comparisons.npmFilesOnlyInLime).toContain(
      "tests/npm-package.test.mjs",
    );
    expect(
      inventory.comparisons.rustSymbolNamesOnlyInLime.length,
    ).toBeGreaterThan(0);
    expect(
      inventory.comparisons.npmSymbolNamesOnlyInLime.length,
    ).toBeGreaterThan(0);
  });

  it("records the independent execpolicy owner", () => {
    expect(inventory.trees["codex-rs/execpolicy"].fileCount).toBeGreaterThan(0);
    expect(inventory.trees["lime-rs/crates/execpolicy"].fileCount).toBeGreaterThan(0);
    expect(inventory.comparisons.execpolicyFilesMissingInLime).toEqual([]);
    const symbols = new Set(
      inventory.trees["lime-rs/crates/execpolicy"].symbols.map(
        (symbol) => symbol.name,
      ),
    );
    for (const name of [
      "ExecPolicyCheckCommand",
      "PolicyParser",
      "Policy",
      "RuleMatch",
      "PatternToken",
      "PrefixPattern",
      "PrefixRule",
    ]) {
      expect(symbols.has(name), name).toBe(true);
    }
    expect(inventory.comparisons.execpolicySymbolNamesMissingInLime).not.toContain(
      "RequirementsExecPolicy",
    );
  });
});
