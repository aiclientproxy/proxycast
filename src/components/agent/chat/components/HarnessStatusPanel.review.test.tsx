import { describe, expect, it } from "vitest";
import {
  renderExpandedPanel as renderPanel,
  getHarnessPanelTestMocks,
} from "./HarnessStatusPanel.testFixtures";

const {
  exportAgentRuntimeReviewDecisionTemplateMock,
  saveAgentRuntimeReviewDecisionMock,
} = getHarnessPanelTestMocks();

describe("HarnessStatusPanel review", () => {
  it("无 App Server current 审核导出通道时不再暴露人工审核 legacy 入口", () => {
    renderPanel({
      diagnosticRuntimeContext: {
        sessionId: "session-review-1",
        workspaceId: "workspace-review-1",
        providerType: "openai",
        model: "gpt-5.4",
        executionStrategy: "react",
        activeTheme: "default",
      },
    });

    expect(
      document.body.querySelector('button[aria-label="导出问题证据包"]'),
    ).toBeNull();
    expect(
      document.body.querySelector('button[aria-label="导出人工审核记录"]'),
    ).toBeNull();
    expect(
      document.body.querySelector('button[aria-label="填写人工审核结果"]'),
    ).toBeNull();
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(exportAgentRuntimeReviewDecisionTemplateMock).not.toHaveBeenCalled();
    expect(saveAgentRuntimeReviewDecisionMock).not.toHaveBeenCalled();
  });
});
