import { readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const SKILLS_EVIDENCE_CURRENT_OWNER_SOURCES = [
  "src/features/capability-drafts/agentEnvelopeDraftPresentation.ts",
  "src/features/capability-drafts/agentEnvelopeDraftPresentation.test.ts",
  "src/features/capability-drafts/workspaceSkillAgentAutomationDraft.ts",
  "src/features/capability-drafts/workspaceSkillAgentAutomationDraft.test.ts",
  "src/features/capability-drafts/components/WorkspaceRegisteredSkillsPanel.tsx",
  "src/features/capability-drafts/components/WorkspaceRegisteredSkillsPanel.test.tsx",
  "src/features/capability-drafts/components/WorkspaceRegisteredSkillsPanel.audit.test.tsx",
  "src/features/capability-drafts/components/WorkspaceRegisteredSkillsPanel.runtime.test.tsx",
  "src/lib/agentRuntime/harnessVerificationPresentation.ts",
  "src/components/skills/SkillsWorkspacePage.tsx",
  "src/components/skills/SkillsWorkspacePageView.tsx",
  "src/components/skills/workspaceSkillRuntimeLaunch.ts",
  "src/components/skills/workspaceSkillRuntimeLaunch.test.ts",
  "src/components/skills/SkillsWorkspacePage.testFixtures.tsx",
] as const;

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), "utf8");
}

describe("skills and evidence current owner boundary", () => {
  it("技能绑定和 evidence consumer 不得回绕 agentRuntime compat 根 barrel", () => {
    for (const relativePath of SKILLS_EVIDENCE_CURRENT_OWNER_SOURCES) {
      const source = readSource(relativePath);

      expect(source, relativePath).not.toContain(
        'from "@/lib/api/agentRuntime"',
      );
      expect(source, relativePath).not.toContain(
        'vi.mock("@/lib/api/agentRuntime"',
      );
    }
  });

  it("技能绑定、evidence 与分域 client 必须保持直接 owner", () => {
    expect(
      readSource(
        "src/features/capability-drafts/components/WorkspaceRegisteredSkillsPanel.tsx",
      ),
    ).toContain('from "@/lib/api/agentRuntime/inventoryClient"');
    expect(
      readSource("src/components/skills/SkillsWorkspacePage.tsx"),
    ).toContain('from "@/lib/api/agentRuntime/toolInventoryTypes"');
    expect(
      readSource("src/lib/agentRuntime/harnessVerificationPresentation.ts"),
    ).toContain('from "@/lib/api/agentRuntime/evidenceTypes"');
  });
});
