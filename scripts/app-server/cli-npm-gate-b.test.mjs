import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  path.resolve(process.cwd(), "scripts/app-server/cli-npm-gate-b.mjs"),
  "utf8",
);
const cliGateSource = readFileSync(
  path.resolve(process.cwd(), "scripts/app-server/cli-gate-b.mjs"),
  "utf8",
);

describe("CLI npm Gate B", () => {
  it("stages the Codex-style npm layout around the real runtime payload", () => {
    expect(source).toContain('"build_npm_package.py"');
    expect(source).toContain('"--vendor-src"');
    expect(source).toContain("`lime${suffix}`");
    expect(source).toContain("`app-server${suffix}`");
    expect(source).toContain("`code-mode-host${suffix}`");
    expect(source).toContain('"windows-sandbox-setup.exe"');
    expect(source).toContain('"windows-sandbox-runner.exe"');
    expect(source).toContain("runtimeLibraries");
  });

  it("enters CLI Gate B through the Node launcher and sibling App Server", () => {
    expect(source).toContain("LIME_CLI_BIN: launcherPath");
    expect(source).toContain('LIME_CLI_GATE_B_USE_SIBLING_APP_SERVER: "1"');
    expect(source).toContain("APP_SERVER_BIN: packagedAppServer");
    expect(source).toContain('"tui-gate-b.mjs"');
    expect(source).toContain('LIME_TUI_GATE_B_SCENARIOS: "complete"');
    expect(cliGateSource).toContain(
      'process.env.LIME_CLI_GATE_B_USE_SIBLING_APP_SERVER !== "1"',
    );
    expect(cliGateSource).toContain("canonical thread identity");
    expect(cliGateSource).toContain("canonical turn identity");
  });
});
