import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const RELIABILITY_CURRENT_OWNER_SOURCES = [
  "src/components/agent/chat/components/AgentThreadFileCheckpointDialog.tsx",
  "src/components/agent/chat/components/AgentThreadPolicyEvidenceCard.tsx",
  "src/components/agent/chat/components/AgentThreadProviderSafetyBufferingCard.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanel.fileCheckpoints.test.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanel.policy.test.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanel.test.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanel.testFixtures.tsx",
  "src/components/agent/chat/components/AgentThreadReliabilityPanelViewModel.ts",
  "src/components/agent/chat/components/AgentThreadReliabilityPanelViewModel.unit.test.ts",
  "src/components/agent/chat/utils/providerSafetyBufferingDiagnostic.ts",
  "src/components/agent/chat/utils/runtimePolicyEvidence.ts",
  "src/components/agent/chat/utils/runtimePolicyEvidence.test.ts",
  "src/components/agent/chat/utils/runtimeRoutingEvidence.ts",
  "src/components/agent/chat/utils/runtimeRoutingEvidence.test.ts",
  "src/components/agent/chat/utils/threadReliabilityDiagnosticText.ts",
  "src/components/agent/chat/utils/threadReliabilityIncidents.ts",
  "src/components/agent/chat/utils/threadReliabilityOutcome.ts",
  "src/components/agent/chat/utils/threadReliabilityRequests.ts",
  "src/components/agent/chat/utils/threadReliabilityStatus.ts",
  "src/components/agent/chat/utils/threadReliabilityTypes.ts",
  "src/components/agent/chat/utils/threadReliabilityView.test.ts",
] as const;

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), "utf8");
}

describe("agent thread reliability current owner boundary", () => {
  it("可靠性 read model 和 file checkpoint 不得回绕 agentRuntime compat 根 barrel", () => {
    for (const relativePath of RELIABILITY_CURRENT_OWNER_SOURCES) {
      const source = readSource(relativePath);

      expect(source, relativePath).not.toContain(
        'from "@/lib/api/agentRuntime"',
      );
      expect(source, relativePath).not.toContain(
        'vi.mock("@/lib/api/agentRuntime"',
      );
    }
  });

  it("DTO 和 file checkpoint 行为必须直接指向各自 current owner", () => {
    expect(
      readSource(
        "src/components/agent/chat/components/AgentThreadReliabilityPanel.tsx",
      ),
    ).toContain('from "@/lib/api/agentRuntime/sessionTypes"');
    expect(
      readSource(
        "src/components/agent/chat/components/AgentThreadFileCheckpointDialog.tsx",
      ),
    ).toContain('from "@/lib/api/agentRuntime/threadClient"');
  });
});
