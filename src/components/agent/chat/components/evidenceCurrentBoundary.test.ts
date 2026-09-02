import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { cwd } from "node:process";
import { describe, expect, it } from "vitest";

const EVIDENCE_CURRENT_OWNER_SOURCES = [
  "src/components/agent/chat/components/HarnessVerificationSummarySection.tsx",
  "src/components/agent/chat/components/RuntimeReviewDecisionDialog.tsx",
  "src/components/agent/chat/components/HarnessStatusPanel.reviewGuard.test.tsx",
  "src/components/agent/chat/utils/curatedTaskRecommendationSignals.ts",
] as const;

const EXPORT_CLIENT_MOCKS = [
  "src/components/agent/chat/components/HarnessStatusPanel.testFixtures.tsx",
] as const;

const DELETED_EVIDENCE_SURFACES = [
  "src/components/agent/chat/components/HarnessEvidencePackCard.tsx",
  "src/components/agent/chat/components/HarnessEvidenceSummarySections.tsx",
  "src/components/agent/chat/components/HarnessHandoffExportSection.tsx",
  "src/components/agent/chat/components/HarnessTaskIndexSection.tsx",
  "src/components/agent/chat/components/harnessEvidencePackStore.ts",
  "src/components/agent/chat/components/harnessEvidenceViewModel.ts",
  "src/components/agent/chat/components/useHarnessEvidencePackExport.ts",
  "src/components/agent/chat/experts/ExpertSkillEvidenceSummary.tsx",
  "src/components/agent/chat/experts/expertSkillEvidenceSummaryViewModel.ts",
  "src/lib/api/agentRuntime/appServerEvidenceExportProjection.ts",
  "src/lib/api/agentRuntime/evidencePackNormalizers.ts",
] as const;

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), "utf8");
}

describe("evidence current owner boundary", () => {
  it("旧 Evidence Pack / Harness surfaces 必须保持物理删除", () => {
    for (const relativePath of DELETED_EVIDENCE_SURFACES) {
      expect(existsSync(join(cwd(), relativePath)), relativePath).toBe(false);
    }
  });

  it("evidence 与 review 消费者不得回绕 agentRuntime compat 根 barrel", () => {
    for (const relativePath of [
      ...EVIDENCE_CURRENT_OWNER_SOURCES,
      ...EXPORT_CLIENT_MOCKS,
    ]) {
      expect(readSource(relativePath), relativePath).not.toContain(
        'from "@/lib/api/agentRuntime"',
      );
      expect(readSource(relativePath), relativePath).not.toContain(
        'vi.mock("@/lib/api/agentRuntime"',
      );
    }
  });

  it("DTO、导出行为与测试 mock 必须直连各自 current owner", () => {
    for (const relativePath of EVIDENCE_CURRENT_OWNER_SOURCES) {
      expect(readSource(relativePath), relativePath).toContain(
        'from "@/lib/api/agentRuntime/evidenceTypes"',
      );
    }

    for (const relativePath of EXPORT_CLIENT_MOCKS) {
      expect(readSource(relativePath), relativePath).toContain(
        'vi.mock("@/lib/api/agentRuntime/exportClient"',
      );
    }
  });
});
