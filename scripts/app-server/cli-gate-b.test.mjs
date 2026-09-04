import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

const source = readFileSync(
  path.resolve(process.cwd(), "scripts/app-server/cli-gate-b.mjs"),
  "utf8",
);
const fixtureSource = readFileSync(
  path.resolve(
    process.cwd(),
    "scripts/app-server/terminal-gate-fixture.mjs",
  ),
  "utf8",
);

describe("CLI Gate B", () => {
  it("runs the real CLI and App Server through the current stdio boundary", () => {
    expect(source).toContain("execFileAsync(cliBinaryPath");
    expect(source).toContain('"--app-server"');
    expect(source).toContain('"--app-server-arg=--backend"');
    expect(source).toContain('"--app-server-arg=external"');
    expect(fixtureSource).toContain('"turn.completed"');
    expect(source).toContain('envelope.result?.status, "ready"');
    expect(source).toContain("canonical thread identity");
    expect(source).toContain("canonical turn identity");
  });

  it("keeps the deterministic fixture out of production backend modes", () => {
    expect(source).not.toContain('"--app-server-arg=mock"');
    expect(source).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(source).not.toContain("setTimeout(");
    expect(source).not.toContain("turn.final_done");
  });
});
