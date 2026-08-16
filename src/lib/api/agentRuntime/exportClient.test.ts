import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createExportClient } from "./exportClient";
import type { AppServerRequestResult } from "@/lib/api/appServer";
import { changeLimeLocale } from "@/i18n/createI18n";
import type { AgentRuntimeExportAppServerClient } from "./exportClient";

const handoffBundleOutput = {
  sessionId: "session-handoff",
  threadId: "thread-handoff",
  workspaceRoot: "/tmp/workspace",
  bundleRelativeRoot: ".lime/harness/sessions/session-handoff",
  bundleAbsoluteRoot: "/tmp/workspace/.lime/harness/sessions/session-handoff",
  exportedAt: "2026-06-08T08:00:00.000Z",
  threadStatus: "running",
  pendingRequestCount: 1,
  queuedTurnCount: 0,
  activeSubagentCount: 2,
  todoTotal: 3,
  todoPending: 1,
  todoInProgress: 1,
  todoCompleted: 1,
  artifacts: [
    {
      kind: "handoff",
      title: "handoff.md",
      relativePath: ".lime/harness/sessions/session-handoff/handoff.md",
      absolutePath:
        "/tmp/workspace/.lime/harness/sessions/session-handoff/handoff.md",
      bytes: 128,
    },
  ],
};

const analysisHandoffOutput = {
  session_id: "session-analysis",
  thread_id: "thread-analysis",
  workspace_root: "/tmp/workspace",
  analysis_relative_root: ".lime/harness/sessions/session-analysis/analysis",
  analysis_absolute_root:
    "/tmp/workspace/.lime/harness/sessions/session-analysis/analysis",
  handoff_bundle_relative_root: ".lime/harness/sessions/session-analysis",
  evidence_pack_relative_root:
    ".lime/harness/sessions/session-analysis/evidence",
  replay_case_relative_root: ".lime/harness/sessions/session-analysis/replay",
  exported_at: "2026-06-08T08:05:00.000Z",
  title: "外部分析",
  thread_status: "waiting_request",
  pending_request_count: 1,
  queued_turn_count: 0,
  sanitized_workspace_root: "/tmp/workspace",
  copy_prompt: "请审阅这份 handoff。",
  artifacts: [
    {
      kind: "analysis_brief",
      title: "analysis-brief.md",
      relative_path:
        ".lime/harness/sessions/session-analysis/analysis/analysis-brief.md",
      absolute_path:
        "/tmp/workspace/.lime/harness/sessions/session-analysis/analysis/analysis-brief.md",
      bytes: 256,
    },
  ],
};

const replayCaseOutput = {
  sessionId: "session-replay",
  threadId: "thread-replay",
  workspaceRoot: "/tmp/workspace",
  replayRelativeRoot: ".lime/harness/sessions/session-replay/replay",
  replayAbsoluteRoot:
    "/tmp/workspace/.lime/harness/sessions/session-replay/replay",
  handoffBundleRelativeRoot: ".lime/harness/sessions/session-replay",
  evidencePackRelativeRoot: ".lime/harness/sessions/session-replay/evidence",
  exportedAt: "2026-06-08T08:10:00.000Z",
  threadStatus: "failed",
  pendingRequestCount: 0,
  queuedTurnCount: 1,
  linkedHandoffArtifactCount: 1,
  linkedEvidenceArtifactCount: 1,
  recentArtifactCount: 2,
  artifacts: [
    {
      kind: "input",
      title: "input.json",
      relativePath: ".lime/harness/sessions/session-replay/replay/input.json",
      absolutePath:
        "/tmp/workspace/.lime/harness/sessions/session-replay/replay/input.json",
      bytes: 512,
    },
  ],
};

const reviewDecision = {
  decisionStatus: "accepted",
  decisionSummary: "确认可以合入。",
  chosenFixStrategy: "保留最小 current 边界。",
  riskLevel: "medium",
  riskTags: ["runtime"],
  humanReviewer: "Lime Maintainer",
  followupActions: ["补 GUI smoke"],
  regressionRequirements: ["npm run test:contracts"],
  notes: "",
};

const reviewDecisionTemplateOutput = {
  sessionId: "session-review",
  threadId: "thread-review",
  workspaceRoot: "/tmp/workspace",
  reviewRelativeRoot: ".lime/harness/sessions/session-review/review",
  reviewAbsoluteRoot:
    "/tmp/workspace/.lime/harness/sessions/session-review/review",
  analysisRelativeRoot: ".lime/harness/sessions/session-review/analysis",
  analysisAbsoluteRoot:
    "/tmp/workspace/.lime/harness/sessions/session-review/analysis",
  handoffBundleRelativeRoot: ".lime/harness/sessions/session-review",
  evidencePackRelativeRoot: ".lime/harness/sessions/session-review/evidence",
  replayCaseRelativeRoot: ".lime/harness/sessions/session-review/replay",
  exportedAt: "2026-06-08T08:15:00.000Z",
  title: "人工审核结论",
  threadStatus: "waiting_request",
  pendingRequestCount: 1,
  queuedTurnCount: 0,
  defaultDecisionStatus: "pending_review",
  decision: reviewDecision,
  decisionStatusOptions: [
    "accepted",
    "deferred",
    "rejected",
    "needs_more_evidence",
    "pending_review",
  ],
  riskLevelOptions: ["low", "medium", "high", "unknown"],
  reviewChecklist: ["确认 evidence pack"],
  analysisArtifacts: [
    {
      kind: "analysis_brief",
      title: "analysis-brief.md",
      relativePath:
        ".lime/harness/sessions/session-review/analysis/analysis-brief.md",
      absolutePath:
        "/tmp/workspace/.lime/harness/sessions/session-review/analysis/analysis-brief.md",
      bytes: 256,
    },
  ],
  artifacts: [
    {
      kind: "review_decision_json",
      title: "review-decision.json",
      relativePath:
        ".lime/harness/sessions/session-review/review/review-decision.json",
      absolutePath:
        "/tmp/workspace/.lime/harness/sessions/session-review/review/review-decision.json",
      bytes: 384,
    },
  ],
};

function appServerClientMock(): AgentRuntimeExportAppServerClient {
  return {
    exportHandoffBundle: vi.fn().mockResolvedValue({
      id: 1,
      result: handoffBundleOutput,
      response: { id: 1, result: handoffBundleOutput },
      notifications: [],
      messages: [],
    }),
    exportAnalysisHandoff: vi.fn().mockResolvedValue({
      id: 1,
      result: analysisHandoffOutput,
      response: { id: 1, result: analysisHandoffOutput },
      notifications: [],
      messages: [],
    }),
    exportReplayCase: vi.fn().mockResolvedValue({
      id: 1,
      result: replayCaseOutput,
      response: { id: 1, result: replayCaseOutput },
      notifications: [],
      messages: [],
    }),
    exportReviewDecisionTemplate: vi.fn().mockResolvedValue({
      id: 1,
      result: reviewDecisionTemplateOutput,
      response: { id: 1, result: reviewDecisionTemplateOutput },
      notifications: [],
      messages: [],
    }),
    saveReviewDecision: vi.fn().mockResolvedValue({
      id: 1,
      result: reviewDecisionTemplateOutput,
      response: { id: 1, result: reviewDecisionTemplateOutput },
      notifications: [],
      messages: [],
    }),
  };
}

function malformedAppServerResult<T>(
  result: unknown,
): AppServerRequestResult<T> {
  return {
    id: 1,
    result: result as T,
    response: {
      id: 1,
      result: result as T,
    },
    notifications: [],
    messages: [],
  };
}

describe("agentRuntime exportClient", () => {
  beforeEach(async () => {
    await changeLimeLocale("en-US");
  });

  afterEach(async () => {
    await changeLimeLocale("zh-CN");
  });

  it("handoff export 应走 App Server current，不回退 legacy command", async () => {
    const appServerClient = appServerClientMock();
    const client = createExportClient({
      appServerClient,
    });

    await expect(
      client.exportAgentRuntimeHandoffBundle(" session-handoff "),
    ).resolves.toMatchObject({
      session_id: "session-handoff",
      active_subagent_count: 2,
      artifacts: [
        expect.objectContaining({
          kind: "handoff",
          relative_path: ".lime/harness/sessions/session-handoff/handoff.md",
        }),
      ],
    });

    expect(appServerClient.exportHandoffBundle).toHaveBeenCalledWith({
      sessionId: "session-handoff",
      locale: "en-US",
    });
  });

  it("analysis / replay / review current 导出应走 App Server 并先校验 DTO 再 normalize", async () => {
    const appServerClient = appServerClientMock();
    const client = createExportClient({
      appServerClient,
    });

    await expect(
      client.exportAgentRuntimeAnalysisHandoff("session-analysis"),
    ).resolves.toMatchObject({
      session_id: "session-analysis",
      copy_prompt: "请审阅这份 handoff。",
      artifacts: [
        expect.objectContaining({
          kind: "analysis_brief",
        }),
      ],
    });
    await expect(
      client.exportAgentRuntimeReplayCase("session-replay"),
    ).resolves.toMatchObject({
      session_id: "session-replay",
      linked_handoff_artifact_count: 1,
      artifacts: [
        expect.objectContaining({
          kind: "input",
        }),
      ],
    });
    await expect(
      client.exportAgentRuntimeReviewDecisionTemplate("session-review"),
    ).resolves.toMatchObject({
      session_id: "session-review",
      decision: expect.objectContaining({
        decision_status: "accepted",
      }),
      artifacts: [
        expect.objectContaining({
          kind: "review_decision_json",
        }),
      ],
    });
    await expect(
      client.saveAgentRuntimeReviewDecision({
        session_id: "session-review",
        decision_status: "accepted",
        decision_summary: "确认可以合入。",
        chosen_fix_strategy: "保留最小 current 边界。",
        risk_level: "medium",
        risk_tags: ["runtime"],
        human_reviewer: "Lime Maintainer",
        followup_actions: ["补 GUI smoke"],
        regression_requirements: ["npm run test:contracts"],
        notes: "",
      }),
    ).resolves.toMatchObject({
      session_id: "session-review",
      decision: expect.objectContaining({
        risk_level: "medium",
      }),
    });

    expect(appServerClient.exportAnalysisHandoff).toHaveBeenCalledWith({
      sessionId: "session-analysis",
      locale: "en-US",
    });
    expect(appServerClient.exportReplayCase).toHaveBeenCalledWith({
      sessionId: "session-replay",
      locale: "en-US",
    });
    expect(appServerClient.exportReviewDecisionTemplate).toHaveBeenCalledWith({
      sessionId: "session-review",
      locale: "en-US",
    });
    expect(appServerClient.saveReviewDecision).toHaveBeenCalledWith({
      sessionId: "session-review",
      decisionStatus: "accepted",
      decisionSummary: "确认可以合入。",
      chosenFixStrategy: "保留最小 current 边界。",
      riskLevel: "medium",
      riskTags: ["runtime"],
      humanReviewer: "Lime Maintainer",
      followupActions: ["补 GUI smoke"],
      regressionRequirements: ["npm run test:contracts"],
      notes: "",
      locale: "en-US",
    });
  });

  it("runtime residual exports 应把 locale 透传给 App Server locale copy service", async () => {
    const appServerClient = appServerClientMock();
    const client = createExportClient({
      appServerClient,
    });

    await client.exportAgentRuntimeHandoffBundle("session-handoff", {
      locale: "en",
    });
    await client.exportAgentRuntimeAnalysisHandoff("session-analysis", {
      locale: "ja",
    });
    await client.exportAgentRuntimeReplayCase("session-replay", {
      locale: "zh-HK",
    });
    await client.exportAgentRuntimeReviewDecisionTemplate("session-review", {
      locale: "ko",
    });
    await client.saveAgentRuntimeReviewDecision({
      session_id: "session-review",
      decision_status: "accepted",
      decision_summary: "确认可以合入。",
      chosen_fix_strategy: "保留最小 current 边界。",
      risk_level: "medium",
      risk_tags: ["runtime"],
      human_reviewer: "Lime Maintainer",
      followup_actions: ["补 GUI smoke"],
      regression_requirements: ["npm run test:contracts"],
      notes: "",
      locale: "en-US",
    });

    expect(appServerClient.exportHandoffBundle).toHaveBeenLastCalledWith({
      sessionId: "session-handoff",
      locale: "en-US",
    });
    expect(appServerClient.exportAnalysisHandoff).toHaveBeenLastCalledWith({
      sessionId: "session-analysis",
      locale: "ja-JP",
    });
    expect(appServerClient.exportReplayCase).toHaveBeenLastCalledWith({
      sessionId: "session-replay",
      locale: "zh-TW",
    });
    expect(
      appServerClient.exportReviewDecisionTemplate,
    ).toHaveBeenLastCalledWith({
      sessionId: "session-review",
      locale: "ko-KR",
    });
    expect(appServerClient.saveReviewDecision).toHaveBeenLastCalledWith(
      expect.objectContaining({
        sessionId: "session-review",
        locale: "en-US",
      }),
    );
  });

  it("handoff current export 收到假成功或缺字段时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.exportHandoffBundle).mockResolvedValueOnce(
      malformedAppServerResult({ success: true }),
    );
    const client = createExportClient({
      appServerClient,
    });

    await expect(
      client.exportAgentRuntimeHandoffBundle("session-handoff"),
    ).rejects.toThrow(
      "agentSession/handoffBundle/export did not return runtime handoff bundle",
    );

    expect(appServerClient.exportHandoffBundle).toHaveBeenCalledWith({
      sessionId: "session-handoff",
      locale: "en-US",
    });
  });

  it("analysis / replay / review current 收到假成功或缺字段时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.exportAnalysisHandoff).mockResolvedValueOnce(
      malformedAppServerResult({
        ...analysisHandoffOutput,
        copy_prompt: undefined,
      }),
    );
    vi.mocked(appServerClient.exportReplayCase).mockResolvedValueOnce(
      malformedAppServerResult({
        ...replayCaseOutput,
        artifacts: [{ kind: "input", title: "input.json" }],
      }),
    );
    vi.mocked(
      appServerClient.exportReviewDecisionTemplate,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...reviewDecisionTemplateOutput,
        decision: {
          ...reviewDecision,
          riskTags: "runtime",
        },
      }),
    );
    vi.mocked(appServerClient.saveReviewDecision).mockResolvedValueOnce(
      malformedAppServerResult({
        error: {
          code: "COMMAND_UNSUPPORTED",
          message: "not available",
        },
      }),
    );
    const client = createExportClient({
      appServerClient,
    });

    await expect(
      client.exportAgentRuntimeAnalysisHandoff("session-analysis"),
    ).rejects.toThrow(
      "agentSession/analysisHandoff/export did not return runtime analysis handoff",
    );
    await expect(
      client.exportAgentRuntimeReplayCase("session-replay"),
    ).rejects.toThrow(
      "agentSession/replayCase/export did not return runtime replay case",
    );
    await expect(
      client.exportAgentRuntimeReviewDecisionTemplate("session-review"),
    ).rejects.toThrow(
      "agentSession/reviewDecisionTemplate/export did not return runtime review decision template",
    );
    await expect(
      client.saveAgentRuntimeReviewDecision({
        session_id: "session-review",
        decision_status: "accepted",
        decision_summary: "确认可以合入。",
        chosen_fix_strategy: "保留最小 current 边界。",
        risk_level: "medium",
        risk_tags: ["runtime"],
        human_reviewer: "Lime Maintainer",
        followup_actions: [],
        regression_requirements: [],
        notes: "",
      }),
    ).rejects.toThrow(
      "agentSession/reviewDecision/save did not return runtime review decision template",
    );
  });

  it("analysis / replay / review current 收到其它 session 的制品路径时应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    vi.mocked(appServerClient.exportAnalysisHandoff).mockResolvedValueOnce(
      malformedAppServerResult({
        ...analysisHandoffOutput,
        evidence_pack_relative_root:
          ".lime/harness/sessions/session-other/evidence",
      }),
    );
    vi.mocked(appServerClient.exportReplayCase).mockResolvedValueOnce(
      malformedAppServerResult({
        ...replayCaseOutput,
        replayRelativeRoot: ".lime/harness/sessions/session-other/replay",
      }),
    );
    vi.mocked(
      appServerClient.exportReviewDecisionTemplate,
    ).mockResolvedValueOnce(
      malformedAppServerResult({
        ...reviewDecisionTemplateOutput,
        sessionId: "session-other",
      }),
    );
    vi.mocked(appServerClient.saveReviewDecision).mockResolvedValueOnce(
      malformedAppServerResult({
        ...reviewDecisionTemplateOutput,
        analysisAbsoluteRoot:
          "/tmp/workspace/.lime/harness/sessions/session-other/analysis",
      }),
    );
    const client = createExportClient({
      appServerClient,
    });

    await expect(
      client.exportAgentRuntimeAnalysisHandoff("session-analysis"),
    ).rejects.toThrow(
      "agentSession/analysisHandoff/export did not return evidencePackRelativeRoot under the requested session",
    );
    await expect(
      client.exportAgentRuntimeReplayCase("session-replay"),
    ).rejects.toThrow(
      "agentSession/replayCase/export did not return replayRelativeRoot under the requested session",
    );
    await expect(
      client.exportAgentRuntimeReviewDecisionTemplate("session-review"),
    ).rejects.toThrow(
      "agentSession/reviewDecisionTemplate/export returned export for a different session",
    );
    await expect(
      client.saveAgentRuntimeReviewDecision({
        session_id: "session-review",
        decision_status: "accepted",
        decision_summary: "确认可以合入。",
        chosen_fix_strategy: "保留最小 current 边界。",
        risk_level: "medium",
        risk_tags: ["runtime"],
        human_reviewer: "Lime Maintainer",
        followup_actions: [],
        regression_requirements: [],
        notes: "",
      }),
    ).rejects.toThrow(
      "agentSession/reviewDecision/save did not return analysisAbsoluteRoot under the requested session",
    );
  });

  it("缺少 sessionId 时 handoff export 应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createExportClient({
      appServerClient,
    });

    await expect(client.exportAgentRuntimeHandoffBundle(" ")).rejects.toThrow(
      "sessionId is required to export App Server handoff bundle",
    );

    expect(appServerClient.exportHandoffBundle).not.toHaveBeenCalled();
  });

  it("缺少 sessionId 时派生导出应 fail closed", async () => {
    const appServerClient = appServerClientMock();
    const client = createExportClient({
      appServerClient,
    });

    await expect(client.exportAgentRuntimeReplayCase(" ")).rejects.toThrow(
      "sessionId is required to export App Server replay case",
    );
    await expect(client.exportAgentRuntimeAnalysisHandoff(" ")).rejects.toThrow(
      "sessionId is required to export App Server analysis handoff",
    );
    await expect(
      client.exportAgentRuntimeReviewDecisionTemplate(" "),
    ).rejects.toThrow(
      "sessionId is required to export App Server review decision template",
    );
    await expect(
      client.saveAgentRuntimeReviewDecision({
        session_id: " ",
        decision_status: "accepted",
        decision_summary: "",
        chosen_fix_strategy: "",
        risk_level: "low",
        risk_tags: [],
        human_reviewer: "",
        followup_actions: [],
        regression_requirements: [],
        notes: "",
      }),
    ).rejects.toThrow(
      "sessionId is required to save App Server review decision",
    );

    expect(appServerClient.exportReplayCase).not.toHaveBeenCalled();
    expect(appServerClient.exportAnalysisHandoff).not.toHaveBeenCalled();
    expect(appServerClient.exportReviewDecisionTemplate).not.toHaveBeenCalled();
    expect(appServerClient.saveReviewDecision).not.toHaveBeenCalled();
  });
});
