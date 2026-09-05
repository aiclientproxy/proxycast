import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  path.resolve(process.cwd(), "scripts/app-server/cli-surface-gate-b.mjs"),
  "utf8",
);

describe("CLI surface Gate B", () => {
  it("covers current Codex-shaped management surfaces through App Server", () => {
    for (const required of [
      '"mcp", "list", "--json"',
      '"features", "list", "--json"',
      '"plugin", "list", "--json"',
      "pluginRead",
      "pluginSearch",
      '["mcp", "start"',
      '["mcp", "stop"',
      '["features", "enable"',
      '["features", "disable"',
      '"queue", "list", "--thread", threadId, "--json"',
      '"debug", "clear-memories"',
      '"debug", "models", "--bundled"',
      '"mcp", "logout", "docs"',
      '"sandbox"',
      '":read-only"',
      "sandbox=stdout|stderr|cwd|exit-code|read-only-fail-closed",
      "oauth-logout=fail-closed",
    ]) {
      expect(source).toContain(required);
    }
    expect(source).toContain('"--app-server"');
    expect(source).toContain('"--app-server-arg=--data-dir"');
    expect(source).toContain("LIME_APP_SERVER_BIN");
    expect(source).toContain("read_only_sandbox_blocks_shell_command");
  });

  it("keeps the fixture on real stdio App Server and does not call providers", () => {
    expect(source).toContain("localAppServerBinaryPath");
    expect(source).not.toContain("mock");
    expect(source).not.toContain("setTimeout(");
    expect(source).toContain('"--app-server-arg=external"');
    expect(source).toContain("writeTerminalExternalBackend");
  });
});
