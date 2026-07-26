import fs from "node:fs";
import { spawnSync } from "node:child_process";

import { describe, expect, it } from "vitest";

const SCRIPT_PATH = "scripts/agent-qc/project-gate-settings-b.mjs";

describe("project Gate SETTINGS-01 Gate B-F CLI guard", () => {
  it("keeps aggregation bound to same-run structured owner evidence", () => {
    const content = fs.readFileSync(SCRIPT_PATH, "utf8");

    expect(content).toContain('".lime"');
    expect(content).toContain('"project-gates"');
    expect(content).toContain(
      "source evidence must be under the same run root",
    );
    expect(content).toContain("buildSettingsGateBFEvidence");
    expect(content).toContain("buildSettingsGateBFailureEvidence");
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });

  it("exposes the supported owner evidence kinds in help", () => {
    const result = spawnSync(process.execPath, [SCRIPT_PATH, "--help"], {
      encoding: "utf8",
    });

    expect(result.status).toBe(0);
    expect(result.stdout).toContain("shell-memory");
    expect(result.stdout).toContain("settings-scenario");
    expect(result.stdout).not.toContain("provider-migration");
  });
});
