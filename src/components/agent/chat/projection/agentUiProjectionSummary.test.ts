import { describe, expect, it } from "vitest";
import type { AgentUiProjectionEvent } from "./agentUiEventProjection";
import {
  formatAgentUiProjectionEventAuxiliaryDetail,
  formatAgentUiProjectionEventDetail,
  summarizeAgentUiSubagentsSurfaceLanes,
  summarizeAgentUiSubagentsSurfaces,
} from "./agentUiProjectionSummary";

describe("agentUiProjectionSummary", () => {
  it("应按 Agent UI v0.6 Subagents surface 聚合专用 lane", () => {
    const events: AgentUiProjectionEvent[] = [
      {
        type: "agent.changed",
        sourceType: "item_completed",
        sequence: 1,
        sessionId: "session-team-1",
        owner: "agent",
        scope: "agent",
        phase: "acting",
        surface: "team_roster",
        persistence: "snapshot",
        runtimeEntity: "subagent_turn",
      },
      {
        type: "agent.spawned",
        sourceType: "item_completed",
        sequence: 2,
        sessionId: "session-team-1",
        agentId: "child-1",
        taskId: "child-1",
        owner: "agent",
        scope: "agent",
        phase: "acting",
        surface: "delegation_graph",
        persistence: "snapshot",
      },
      {
        type: "worker.notification",
        sourceType: "item_completed",
        sequence: 3,
        sessionId: "session-team-1",
        agentId: "child-1",
        taskId: "child-1",
        owner: "agent",
        scope: "agent",
        phase: "completed",
        surface: "worker_notifications",
        persistence: "archive",
      },
      {
        type: "review.completed",
        sourceType: "evidence_projection",
        sequence: 5,
        sessionId: "session-team-1",
        reviewId: "review-1",
        owner: "evidence",
        scope: "evidence",
        phase: "completed",
        surface: "review_lane",
        persistence: "evidence_pack",
      },
      {
        type: "tool.started",
        sourceType: "tool_start",
        sequence: 6,
        sessionId: "session-team-1",
        owner: "tool",
        scope: "tool_call",
        phase: "acting",
        surface: "tool_ui",
        persistence: "transcript",
      },
    ];

    const lanes = summarizeAgentUiSubagentsSurfaceLanes(events);

    expect(lanes.map((lane) => lane.id)).toEqual([
      "team-topology",
      "worker-flow",
      "review-handoff",
    ]);
    expect(lanes.find((lane) => lane.id === "team-topology")?.total).toBe(2);
    expect(lanes.find((lane) => lane.id === "worker-flow")?.total).toBe(1);
    expect(lanes.find((lane) => lane.id === "review-handoff")?.total).toBe(1);
    expect(
      lanes.find((lane) => lane.id === "worker-flow")?.latestEvents[0]
        ?.sequence,
    ).toBe(3);
  });

  it("应按单个 Subagents surface 产出可交互详情 selector", () => {
    const events: AgentUiProjectionEvent[] = [
      {
        type: "review.requested",
        sourceType: "evidence_projection",
        sequence: 1,
        sessionId: "session-team-1",
        reviewId: "review-1",
        owner: "evidence",
        scope: "evidence",
        phase: "reviewing",
        surface: "review_lane",
        persistence: "evidence_pack",
      },
      {
        type: "task.changed",
        sourceType: "team_control_projection",
        sequence: 2,
        sessionId: "session-team-1",
        reviewId: "review-1",
        taskId: "review-1",
        owner: "task",
        scope: "task",
        phase: "reviewing",
        surface: "review_lane",
        persistence: "snapshot",
        control: "request_review",
        runtimeEntity: "work_item",
      },
      {
        type: "task.changed",
        sourceType: "team_control_projection",
        sequence: 3,
        sessionId: "session-team-1",
        owner: "task",
        scope: "task",
        phase: "acting",
        surface: "team_policy",
        persistence: "snapshot",
      },
    ];

    const surfaces = summarizeAgentUiSubagentsSurfaces(events, {
      latestLimit: 1,
    });

    expect(surfaces.map((surface) => surface.surface)).toEqual([
      "review_lane",
      "team_policy",
    ]);
    expect(surfaces[0]).toMatchObject({
      label: "Review",
      total: 2,
      latestEvents: [expect.objectContaining({ sequence: 2 })],
    });
  });

  it("应为 delegated plan approval 展示请求方与目标", () => {
    const requiredEvent: AgentUiProjectionEvent = {
      type: "action.required",
      sourceType: "tool_progress",
      sequence: 1,
      sessionId: "session-team-1",
      actionId: "approval-1",
      owner: "action",
      scope: "action_request",
      phase: "waiting",
      surface: "hitl",
      persistence: "snapshot",
      control: "approve",
      payload: {
        actionType: "plan_approval",
        decisionKind: "plan_approval_request",
        from: "child-session-1",
        deliveryTarget: "team-lead",
        awaitingLeaderApproval: true,
        planFilePath: ".lime/plans/child-session-1.md",
      },
    };
    const resolvedEvent: AgentUiProjectionEvent = {
      type: "action.resolved",
      sourceType: "action_resolved",
      sequence: 2,
      sessionId: "session-team-1",
      actionId: "approval-1",
      owner: "action",
      scope: "action_request",
      phase: "completed",
      surface: "hitl",
      persistence: "snapshot",
      control: "approve",
      payload: {
        actionType: "plan_approval",
        decisionKind: "plan_approval_response",
        approved: true,
        permissionMode: "ask",
        targetSessionId: "child-session-1",
        planFile: ".lime/plans/child-session-1.md",
        planId: "plan-1",
      },
    };

    expect(formatAgentUiProjectionEventDetail(requiredEvent)).toBe(
      "plan_approval_request",
    );
    expect(formatAgentUiProjectionEventAuxiliaryDetail(requiredEvent)).toBe(
      "决策：plan_approval_request / 请求方：child-session-1 / 投递：team-lead / 等待 leader 审批 / 计划：.lime/plans/child-session-1.md",
    );
    expect(formatAgentUiProjectionEventDetail(resolvedEvent)).toBe(
      "plan_approval_response",
    );
    expect(formatAgentUiProjectionEventAuxiliaryDetail(resolvedEvent)).toBe(
      "决策：plan_approval_response / 目标：child-session-1 / 权限：ask / 结果：已批准 / 计划：.lime/plans/child-session-1.md / Plan：plan-1",
    );
  });

  it("应为 review lane 产出审核细状态辅助详情", () => {
    expect(
      formatAgentUiProjectionEventAuxiliaryDetail({
        type: "review.completed",
        sourceType: "evidence_projection",
        sequence: 1,
        sessionId: "session-team-1",
        reviewId: "review-1",
        owner: "evidence",
        scope: "evidence",
        phase: "completed",
        surface: "review_lane",
        persistence: "evidence_pack",
        payload: {
          decisionStatus: "rejected",
          reviewer: "Lime Maintainer",
          riskLevel: "medium",
          checklistCount: 2,
          followupActionCount: 1,
          regressionRequirementCount: 2,
          requestedFixes: ["补齐权限确认证据", "重新导出 evidence pack"],
          regressionRequirements: ["npm run test:contracts"],
        },
      }),
    ).toBe(
      "决策：rejected / 审核人：Lime Maintainer / 风险：medium / 清单 2 / 后续 1 / 回归 2 / 修复：补齐权限确认证据 +1 / 回归项：npm run test:contracts",
    );
  });

  it("应为 work board reassignment 产出结构化详情", () => {
    const event: AgentUiProjectionEvent = {
      type: "task.changed",
      sourceType: "team_control_projection",
      sequence: 1,
      sessionId: "session-team-1",
      taskId: "work-item-1",
      workItemId: "work-item-1",
      owner: "task",
      scope: "task",
      phase: "routing",
      surface: "work_board",
      persistence: "snapshot",
      control: "assign",
      runtimeEntity: "work_item",
      payload: {
        taskEvent: "team_reassignment",
        action: "reassign",
        previousAssigneeId: "researcher",
        nextAssigneeId: "implementer",
        reassignmentReason: "实现阶段需要切换负责人",
      },
    };

    expect(formatAgentUiProjectionEventDetail(event)).toBe(
      "重新指派：work-item-1",
    );
    expect(formatAgentUiProjectionEventAuxiliaryDetail(event)).toBe(
      "负责人：researcher → implementer / 原因：实现阶段需要切换负责人",
    );
  });

  it("应为 handoff lane 产出 specialist handoff 状态详情", () => {
    expect(
      formatAgentUiProjectionEventAuxiliaryDetail({
        type: "agent.handoff",
        sourceType: "evidence_projection",
        sequence: 1,
        sessionId: "session-team-1",
        handoffId: "handoff-1",
        owner: "agent",
        scope: "agent",
        phase: "accepted",
        surface: "handoff_lane",
        persistence: "evidence_pack",
        topology: "specialist_handoff",
        payload: {
          handoffEvent: "specialist_handoff",
          status: "accepted",
          from: "coordinator",
          to: "specialist",
          resumeTarget: "agent-runtime://session/session-specialist",
          contextBoundary: "workspace_root",
        },
      }),
    ).toBe(
      "状态：accepted / 交接：coordinator → specialist / 恢复：agent-runtime://session/session-specialist / 边界：workspace_root",
    );
  });
});
