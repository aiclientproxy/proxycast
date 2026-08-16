import { describe, expect, it } from "vitest";
import {
  getHarnessPanelTestMocks,
  renderExpandedPanel as renderPanel,
} from "./HarnessStatusPanel.testFixtures";

const {
  exportAgentRuntimeAnalysisHandoffMock,
  exportAgentRuntimeHandoffBundleMock,
  exportAgentRuntimeReplayCaseMock,
  exportAgentRuntimeReviewDecisionTemplateMock,
  saveAgentRuntimeReviewDecisionMock,
} = getHarnessPanelTestMocks();

describe("HarnessStatusPanel current runtime facts", () => {
  it("有 sessionId 但没有 canonical thread/read 时不暴露派生导出入口", () => {
    renderPanel({
      diagnosticRuntimeContext: {
        sessionId: "session-current-1",
        workspaceId: "workspace-current-1",
        providerType: "openai",
        model: "gpt-5.4",
        executionStrategy: "react",
        activeTheme: "default",
      },
    });

    expect(document.body.textContent).not.toContain("问题证据包");
    expect(document.body.textContent).not.toContain("导出问题证据");
    expect(
      document.body.querySelector('button[aria-label="导出问题证据包"]'),
    ).toBeNull();
    expect(exportAgentRuntimeHandoffBundleMock).not.toHaveBeenCalled();
    expect(exportAgentRuntimeReplayCaseMock).not.toHaveBeenCalled();
    expect(exportAgentRuntimeAnalysisHandoffMock).not.toHaveBeenCalled();
    expect(exportAgentRuntimeReviewDecisionTemplateMock).not.toHaveBeenCalled();
    expect(saveAgentRuntimeReviewDecisionMock).not.toHaveBeenCalled();
  });

  it("canonical thread/read facts 直接进入运行时事实区块", () => {
    renderPanel({
      layout: "dialog",
      diagnosticRuntimeContext: {
        sessionId: "session-current-2",
        workspaceId: "workspace-current-2",
        providerType: "openai",
        model: "gpt-5.4",
        executionStrategy: "react",
        activeTheme: "default",
      },
      threadRead: {
        thread_id: "thread-current-2",
        status: "completed",
        turns: [
          { turn_id: "turn-1", status: "completed" },
          { turn_id: "turn-2", status: "completed" },
        ],
        thread_items: [],
        pending_requests: [
          { id: "request-1", request_type: "tool", status: "pending" },
        ],
        artifacts: [{ id: "artifact-1" }],
        evidence_summary: { evidence_refs: ["thread://thread-current-2"] },
      } as never,
    });

    expect(document.body.textContent).toContain("运行时事实");
    expect(document.body.textContent).toContain("thread-current-2");
    expect(document.body.textContent).toContain("completed");
    expect(document.body.textContent).toContain("2 / 0");
    expect(document.body.textContent).toContain("Artifact / Evidence 引用");
    expect(document.body.textContent).not.toContain("问题证据包");
  });
});
