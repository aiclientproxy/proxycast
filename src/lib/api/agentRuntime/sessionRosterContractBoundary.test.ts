import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const CONTRACT_SOURCES = [
  "src/lib/api/agentRuntime/sessionTypes.ts",
  "src/lib/api/agentRuntime/sessionNormalizers.ts",
  "src/lib/api/agentRuntime/normalizers.ts",
  "src/lib/api/agentRuntime/normalizers.d.ts",
  "src/components/agent/chat/projection/canonicalChildThreadSummary.ts",
  "scripts/agent-runtime/tool-surface-page-smoke.mjs",
] as const;

function readSource(relativePath: string): string {
  return readFileSync(join(process.cwd(), relativePath), "utf8");
}

describe("retired session roster contract boundary", () => {
  it("DTO、normalizer 与 fixture 不得恢复 legacy roster", () => {
    const combined = CONTRACT_SOURCES.map(readSource).join("\n");

    expect(combined).not.toContain("AgentSubagentSessionInfo");
    expect(combined).not.toContain("AgentSubagentParentContext");
    expect(combined).not.toContain("AgentSubagentSkillInfo");
    expect(combined).not.toContain("normalizeSubagentSessionInfo");
    expect(combined).not.toContain("normalizeSubagentParentContext");
    expect(combined).not.toContain("child_subagent_sessions");
    expect(combined).not.toContain("sibling_subagent_sessions");
    expect(combined).not.toContain("subagent_parent_context");
  });

  it("两个 production session read 边界不得包含 retired key 或 sanitizer", () => {
    for (const sourcePath of [
      "src/lib/api/agentRuntime/appServerSessionClient.ts",
      "src/lib/api/agentRuntime/sessionClient.ts",
    ]) {
      const source = readSource(sourcePath);

      expect(source).not.toContain("child_subagent_sessions");
      expect(source).not.toContain("subagent_parent_context");
      expect(source).not.toContain("omitLegacyRosterFields");
    }
  });
});
