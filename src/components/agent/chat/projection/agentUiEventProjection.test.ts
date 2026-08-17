import { describe, expect, it } from "vitest";
import type { AgentEvent } from "@/lib/api/agentProtocol";
import {
  buildAgentUiEvidenceChangedEvent,
  buildAgentUiHandoffProjectionEvents,
  buildAgentUiMetricChangedEvent,
  buildAgentUiProjectionEvents,
  buildAgentUiRemoteTeammateProjectionEvents,
  buildAgentUiReviewProjectionEvents,
  buildAgentUiTeamControlProjectionEvents,
} from "./agentUiEventProjection";

const baseContext = {
  sequence: 10,
  timestamp: "2026-05-09T00:00:00.000Z",
  sessionId: "session-1",
  runId: "agent_turn_stream:session-1",
  messageId: "assistant-1",
};

describe("agentUiEventProjection", () => {
  it("应把文本、推理与运行状态映射到标准 Agent UI envelope", () => {
    expect(
      buildAgentUiProjectionEvents(
        { type: "text_delta", text: "最终答案" },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "text.delta",
      sourceType: "text_delta",
      sequence: 10,
      sessionId: "session-1",
      runId: "agent_turn_stream:session-1",
      messageId: "assistant-1",
      owner: "model",
      scope: "part",
      phase: "producing",
      surface: "conversation",
      persistence: "transcript",
      payload: {
        textLength: 4,
        preview: "最终答案",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        { type: "thinking_delta", text: "先分析" },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "reasoning.delta",
      owner: "model",
      phase: "reasoning",
      surface: "inline_process",
      persistence: "ephemeral_live",
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "reasoning_summary_delta",
          itemId: "reasoning-1",
          reasoningId: "reasoning-1",
          summaryIndex: 0,
          text: "先分析",
          delta: "先分析",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "reasoning.delta",
      sourceType: "reasoning_summary_delta",
      partId: "reasoning-1",
      payload: {
        streamKind: "summary",
        summaryIndex: 0,
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "reasoning_content_delta",
          itemId: "reasoning-1",
          reasoningId: "reasoning-1",
          contentIndex: 0,
          text: "raw",
          delta: "raw",
        },
        baseContext,
      ),
    ).toEqual([]);

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_completed",
          item: {
            id: "reasoning-1",
            thread_id: "session-1",
            turn_id: "turn-1",
            sequence: 2,
            status: "completed",
            started_at: "2026-05-09T00:00:00.000Z",
            completed_at: "2026-05-09T00:00:01.000Z",
            updated_at: "2026-05-09T00:00:01.000Z",
            type: "reasoning",
            text: "先分析",
            summary: ["先分析"],
            content: ["raw"],
          },
        },
        baseContext,
      ).map((event) => event.type),
    ).toEqual(["reasoning.summary"]);

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "reasoning_final",
          reasoningId: "runtime-thinking",
          text: "先分析完整过程",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "reasoning.delta",
      sourceType: "reasoning_final",
      owner: "model",
      phase: "reasoning",
      surface: "inline_process",
      persistence: "ephemeral_live",
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "runtime_status",
          status: {
            phase: "permission_review",
            title: "等待确认",
            detail: "需要批准工具执行",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      owner: "runtime",
      scope: "run",
      phase: "waiting",
      surface: "runtime_status",
      payload: {
        title: "等待确认",
        sourcePhase: "permission_review",
      },
    });
  });

  it("应把 model.effective 映射为模型生效事件", () => {
    expect(
      buildAgentUiProjectionEvents(
        {
          type: "model_effective",
          modelRef: {
            providerId: "openai",
            modelId: "gpt-codex",
          },
          modelName: "gpt-codex",
          serviceModelSlot: "coding",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      sourceType: "model_effective",
      sequence: 10,
      owner: "runtime",
      scope: "run",
      payload: {
        model: "gpt-codex",
        mode: "coding",
      },
    });
  });

  it("应把 runtime permission metadata 映射为 permission.changed", () => {
    const events = buildAgentUiProjectionEvents(
      {
        type: "runtime_status",
        status: {
          phase: "permission_review",
          title: "等待权限确认",
          detail: "需要批准 profile",
          metadata: {
            permission_status: "requires_confirmation",
            required_profile_keys: ["read_files", "write_artifacts"],
            ask_profile_keys: ["read_files"],
            blocking_profile_keys: [],
            decision_source: "runtime",
            decision_scope: "turn",
            confirmation_status: "not_requested",
            confirmation_request_id: "approval-1",
            confirmation_source: "policy",
          },
        },
      },
      baseContext,
    );

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "run.status",
      sequence: 10,
    });
    expect(events[1]).toMatchObject({
      type: "permission.changed",
      sourceType: "runtime_status",
      sequence: 11,
      actionId: "approval-1",
      owner: "policy",
      scope: "run",
      phase: "waiting",
      surface: "hitl",
      persistence: "snapshot",
      control: "approve",
      payload: {
        permissionStatus: "requires_confirmation",
        confirmationStatus: "not_requested",
        confirmationRequestId: "approval-1",
        confirmationSource: "policy",
        decisionSource: "runtime",
        decisionScope: "turn",
        requiredProfileKeys: ["read_files", "write_artifacts"],
        askProfileKeys: ["read_files"],
        blockingProfileKeys: [],
        sourcePhase: "permission_review",
      },
    });
  });

  it("应把 runtimeEntity 与 team queue facts 写入 v0.6 标准 envelope", () => {
    const events = buildAgentUiProjectionEvents(
      {
        type: "runtime_status",
        status: {
          phase: "routing",
          title: "等待 provider 并发",
          detail: "openai 队列已满",
          metadata: {
            team_phase: "queued",
            team_parallel_budget: 2,
            team_active_count: 1,
            team_queued_count: 1,
            provider_concurrency_group: "openai",
            provider_parallel_budget: 3,
            queue_reason: "provider_busy",
            retryable_overload: true,
          },
        },
      },
      baseContext,
    );

    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      type: "run.status",
      runtimeEntity: "agent_turn",
      runtimeStatus: "preparing",
      latestTurnStatus: "preparing",
      teamPhase: "queued",
      teamParallelBudget: 2,
      teamActiveCount: 1,
      teamQueuedCount: 1,
      queuedTurnCount: 1,
      providerConcurrencyGroup: "openai",
      providerParallelBudget: 3,
      queueReason: "provider_busy",
      retryableOverload: true,
      payload: {
        runtimeEntity: "agent_turn",
        teamPhase: "queued",
        teamParallelBudget: 2,
        teamActiveCount: 1,
        teamQueuedCount: 1,
        providerConcurrencyGroup: "openai",
      },
    });
  });

  it("应把显式 remote task 投影为 remote teammate 与 external task", () => {
    const running = buildAgentUiRemoteTeammateProjectionEvents(
      {
        event: "needs_input",
        remoteTaskId: "remote-task-1",
        agentName: "远端审阅者",
        agentCardId: "agent-card-1",
        agentCardUrl: "https://remote.example/.well-known/agent-card.json",
        provider: "a2a",
        summaryPreview: "等待远端补充授权",
        inputRequired: true,
        authRequired: true,
        artifactIds: ["artifact-1"],
      },
      baseContext,
    );

    expect(running).toContainEqual(
      expect.objectContaining({
        type: "agent.changed",
        sourceType: "remote_task_projection",
        sessionId: "session-1",
        taskId: "remote-task-1",
        agentId: "agent-card-1",
        agentName: "远端审阅者",
        agentRole: "remote_teammate",
        remoteTaskId: "remote-task-1",
        phase: "waiting",
        surface: "remote_teammate",
        control: "answer",
        topology: "remote_teammate",
        runtimeEntity: "external_task",
        runtimeStatus: "needs_input",
        payload: expect.objectContaining({
          remoteEvent: "needs_input",
          agentCardId: "agent-card-1",
          provider: "a2a",
          inputRequired: true,
          authRequired: true,
        }),
        refs: {
          artifactIds: ["artifact-1"],
        },
      }),
    );
    expect(running).toContainEqual(
      expect.objectContaining({
        type: "task.changed",
        sourceType: "remote_task_projection",
        taskId: "remote-task-1",
        surface: "remote_teammate",
        control: "answer",
        runtimeEntity: "external_task",
        runtimeStatus: "needs_input",
      }),
    );
    expect(running).toContainEqual(
      expect.objectContaining({
        type: "artifact.updated",
        sourceType: "remote_task_projection",
        artifactId: "artifact-1",
        surface: "artifact_workspace",
        runtimeEntity: "external_task",
      }),
    );

    const completed = buildAgentUiRemoteTeammateProjectionEvents(
      {
        event: "completed",
        remoteTaskId: "remote-task-1",
        provider: "a2a",
        status: "completed",
      },
      baseContext,
    );

    expect(completed).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "agent.changed",
          runtimeEntity: "external_task",
          runtimeStatus: "completed",
          surface: "remote_teammate",
          persistence: "snapshot",
        }),
        expect.objectContaining({
          type: "task.changed",
          runtimeEntity: "external_task",
          runtimeStatus: "completed",
          surface: "remote_teammate",
          persistence: "snapshot",
        }),
        expect.objectContaining({
          type: "worker.notification",
          workerNotificationId: "remote-task-1:completed",
          runtimeEntity: "external_task",
          runtimeStatus: "completed",
          surface: "worker_notifications",
          payload: expect.objectContaining({
            notificationKind: "remote_task_terminal",
          }),
        }),
      ]),
    );
  });

  it("应把 auth_required remote task 作为真实待输入状态投影", () => {
    const events = buildAgentUiRemoteTeammateProjectionEvents(
      {
        event: "auth_required",
        remoteTaskId: "remote-task-auth",
        provider: "a2a",
      },
      baseContext,
    );

    expect(events).toContainEqual(
      expect.objectContaining({
        type: "agent.changed",
        taskId: "remote-task-auth",
        control: "answer",
        runtimeEntity: "external_task",
        runtimeStatus: "needs_input",
        surface: "remote_teammate",
        payload: expect.objectContaining({
          remoteEvent: "auth_required",
          inputRequired: true,
          authRequired: true,
          authStatus: "auth_required",
        }),
      }),
    );
  });

  it("应把 Team controls adapter 投影到 work board 与标准 control 语义", () => {
    const teamThreadContext = {
      ...baseContext,
      threadId: "thread-team-1",
      turnId: "turn-parent-1",
    };
    const events = buildAgentUiTeamControlProjectionEvents(
      {
        action: "resume",
        sessionId: "session-team-1",
        requestedSessionIds: ["child-1", "child-1"],
        affectedSessionIds: ["child-1"],
        cascadeSessionIds: ["child-2"],
      },
      teamThreadContext,
    );

    expect(events).toHaveLength(2);
    expect(events.every((event) => event.threadId === "thread-team-1")).toBe(
      true,
    );
    expect(events[0]).toMatchObject({
      type: "task.changed",
      sourceType: "team_control_projection",
      sequence: 10,
      taskId: "child-1",
      agentId: "child-1",
      owner: "task",
      scope: "task",
      phase: "acting",
      surface: "work_board",
      control: "continue_agent",
      runtimeEntity: "subagent_turn",
      runtimeStatus: "running",
      latestTurnStatus: "running",
      payload: {
        taskEvent: "team_control",
        action: "resume",
        taskId: "child-1",
        runtimeEntity: "subagent_turn",
      },
    });
    expect(events[1]).toMatchObject({
      type: "agent.handoff",
      sourceType: "team_control_projection",
      sequence: 11,
      sessionId: "session-team-1",
      threadId: "thread-team-1",
      turnId: "turn-parent-1",
      taskId: "child-1",
      agentId: "child-1",
      handoffId: "session-team-1:handoff:child-1",
      parentThreadId: "thread-team-1",
      owner: "agent",
      scope: "agent",
      phase: "completed",
      surface: "handoff_lane",
      control: "continue_agent",
      topology: "specialist_handoff",
      runtimeEntity: "subagent_turn",
      runtimeStatus: "running",
      payload: {
        handoffEvent: "specialist_handoff",
        status: "resumed",
        sourceControl: "resume",
        from: "session-team-1",
        to: "child-1",
        reason: "team_control_resume",
        resumeTarget: "agent-runtime://session/child-1",
        contextBoundary: "subagent_session",
      },
    });

    expect(
      buildAgentUiTeamControlProjectionEvents(
        {
          action: "wait",
          requestedSessionIds: ["child-1"],
          timedOut: true,
        },
        baseContext,
      )[0],
    ).toMatchObject({
      control: "wait",
      phase: "waiting",
      payload: {
        timedOut: true,
      },
    });

    expect(
      buildAgentUiTeamControlProjectionEvents(
        {
          action: "delegate",
          requestedSessionIds: ["child-delegated"],
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "task.changed",
      control: "delegate",
      phase: "planning",
      surface: "delegation_graph",
      runtimeEntity: "subagent_turn",
      runtimeStatus: "queued",
    });

    expect(
      buildAgentUiTeamControlProjectionEvents(
        {
          action: "assign",
          requestedSessionIds: ["work-item-1"],
          workItemId: "work-item-1",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "task.changed",
      control: "assign",
      phase: "planning",
      surface: "work_board",
      runtimeEntity: "work_item",
      workItemId: "work-item-1",
      payload: {
        runtimeEntity: "work_item",
        workItemId: "work-item-1",
      },
    });

    expect(
      buildAgentUiTeamControlProjectionEvents(
        {
          action: "reassign",
          requestedSessionIds: ["work-item-2"],
          workItemId: "work-item-2",
          previousAssigneeId: "researcher",
          nextAssigneeId: "implementer",
          reassignmentReason: "实现阶段需要切换负责人",
          resolvedStatus: "assigned",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "task.changed",
      taskId: "work-item-2",
      control: "assign",
      phase: "accepted",
      surface: "work_board",
      runtimeEntity: "work_item",
      runtimeStatus: "accepted",
      workItemId: "work-item-2",
      payload: {
        taskEvent: "team_reassignment",
        action: "reassign",
        resolvedStatus: "assigned",
        previousAssigneeId: "researcher",
        nextAssigneeId: "implementer",
        reassignmentReason: "实现阶段需要切换负责人",
      },
    });

    expect(
      buildAgentUiTeamControlProjectionEvents(
        {
          action: "request_review",
          requestedSessionIds: [],
          workItemId: "review-1",
          reviewId: "review-1",
          runtimeEntity: "work_item",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "task.changed",
      taskId: "review-1",
      control: "request_review",
      phase: "reviewing",
      surface: "review_lane",
      reviewId: "review-1",
      workItemId: "review-1",
      runtimeEntity: "work_item",
      runtimeStatus: "waiting",
    });
  });

  it("应把工具输入、成功输出和失败输出映射为 tool lifecycle", () => {
    const started = buildAgentUiProjectionEvents(
      {
        type: "tool_start",
        tool_id: "tool-1",
        tool_name: "read_file",
        arguments: JSON.stringify({ path: "README.md" }),
      },
      baseContext,
    );

    expect(started).toHaveLength(2);
    expect(started[0]).toMatchObject({
      type: "tool.started",
      sequence: 10,
      toolCallId: "tool-1",
      owner: "tool",
      phase: "acting",
    });
    expect(started[1]).toMatchObject({
      type: "tool.args",
      sequence: 11,
      payload: {
        toolName: "read_file",
        inputAvailable: true,
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "tool_input_delta",
          tool_id: "tool-1",
          tool_name: "read_file",
          delta: '{"path"',
          accumulated_arguments: '{"path"',
          provider: "openai_compatible",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.args.delta",
      sourceType: "tool_input_delta",
      toolCallId: "tool-1",
      phase: "acting",
      payload: {
        toolName: "read_file",
        provider: "openai_compatible",
        inputStreaming: true,
        deltaPreview: '{"path"',
        deltaLength: 7,
        accumulatedInputLength: 7,
      },
    });

    const result = buildAgentUiProjectionEvents(
      {
        type: "tool_end",
        tool_id: "tool-1",
        result: {
          success: true,
          output: "已读取文件",
          metadata: {
            artifact_id: "artifact-1",
            artifact_path: ".lime/artifacts/demo.md",
          },
        },
      },
      baseContext,
    )[0];

    expect(result).toMatchObject({
      type: "tool.result",
      toolCallId: "tool-1",
      phase: "completed",
      persistence: "archive",
      refs: {
        artifactIds: ["artifact-1"],
        artifactPaths: [".lime/artifacts/demo.md"],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "tool_end",
          tool_id: "tool-2",
          result: {
            success: false,
            output: "",
            error: "权限不足",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.failed",
      toolCallId: "tool-2",
      phase: "failed",
      payload: {
        success: false,
        errorPreview: "权限不足",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "tool_progress",
          tool_id: "tool-1",
          progress: {
            message: "正在处理第 2 项",
            progress: 2,
            total: 4,
            metadata: {
              notification_kind: "mcp_progress",
            },
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.progress",
      toolCallId: "tool-1",
      phase: "acting",
      persistence: "ephemeral_live",
      payload: {
        messagePreview: "正在处理第 2 项",
        progress: 2,
        total: 4,
        metadataKeys: ["notification_kind"],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "tool_output_delta",
          tool_id: "tool-1",
          delta: "partial output",
          output_kind: "log",
          metadata: {
            notification_kind: "mcp_log",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.output.delta",
      toolCallId: "tool-1",
      phase: "acting",
      payload: {
        outputKind: "log",
        deltaPreview: "partial output",
        deltaLength: 14,
        metadataKeys: ["notification_kind"],
      },
    });
  });

  it("应把 action、artifact 和 queue 映射到独立 taxonomy", () => {
    expect(
      buildAgentUiProjectionEvents(
        {
          type: "action_required",
          request_id: "approval-1",
          action_type: "tool_confirmation",
          prompt: "允许执行命令？",
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "action.required",
      actionId: "approval-1",
      owner: "action",
      scope: "action_request",
      phase: "waiting",
      surface: "hitl",
      control: "approve",
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "artifact_snapshot",
          artifact: {
            artifactId: "artifact-2",
            filePath: ".lime/artifacts/report.md",
            content: "# 报告",
            metadata: {
              complete: false,
            },
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "artifact.updated",
      artifactId: "artifact-2",
      owner: "artifact",
      phase: "producing",
      surface: "artifact_workspace",
      refs: {
        artifactIds: ["artifact-2"],
        artifactPaths: [".lime/artifacts/report.md"],
      },
    });
  });

  it.each(["queue_added", "queue_removed", "queue_started", "queue_cleared"])(
    "已退役的 raw user-turn queue 事件 %s 不应进入 Agent UI 投影",
    (type) => {
      expect(
        buildAgentUiProjectionEvents(
          { type } as unknown as AgentEvent,
          baseContext,
        ),
      ).toEqual([]);
    },
  );

  it("已退役的 raw subagent status 不应进入 Agent UI 投影", () => {
    expect(
      buildAgentUiProjectionEvents(
        {
          type: "subagent_status_changed",
          session_id: "child-1",
          root_session_id: "session-1",
          status: "running",
        } as unknown as AgentEvent,
        baseContext,
      ),
    ).toEqual([]);
  });

  it("应从 artifact metadata 的 requested fix 执行结果即时回写 work_board", () => {
    const events = buildAgentUiProjectionEvents(
      {
        type: "artifact_snapshot",
        artifact: {
          artifactId: "artifact-fix-execution-1",
          filePath: ".lime/harness/sessions/session-1/review/fix-result.json",
          content: "{}",
          metadata: {
            complete: true,
            reviewId: "review/root",
            requestedFixExecutionResults: [
              {
                requestedFix: "补一条 release note",
                requestedFixIndex: 1,
                executionStatus: "completed",
                regressionOutcome: "recovered",
                summaryPreview: "release note 已补齐并完成回归。",
                resultRef:
                  "agent-runtime://session/session-1/thread/thread-1/turn/turn-review/item/item-fix-1",
                artifactIds: ["artifact-release-note"],
                artifactPaths: ["docs/release-note.md"],
              },
            ],
          },
        },
      },
      baseContext,
    );

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "artifact.preview.ready",
      surface: "artifact_workspace",
    });
    expect(events[1]).toMatchObject({
      type: "task.changed",
      sourceType: "artifact_snapshot",
      taskId: "review/root:requested-fix:1",
      workItemId: "review/root:requested-fix:1",
      reviewId: "review/root",
      artifactId: "artifact-fix-execution-1",
      owner: "task",
      scope: "task",
      phase: "completed",
      surface: "work_board",
      persistence: "snapshot",
      control: "open_detail",
      topology: "review_team",
      runtimeEntity: "work_item",
      runtimeStatus: "completed",
      payload: {
        taskEvent: "review_requested_fix",
        executionSource: "artifact_snapshot_metadata",
        requestedFix: "补一条 release note",
        requestedFixIndex: 1,
        executionStatus: "completed",
        regressionOutcome: "recovered",
        executionSummaryPreview: "release note 已补齐并完成回归。",
        executionResultRef:
          "agent-runtime://session/session-1/thread/thread-1/turn/turn-review/item/item-fix-1",
        executionArtifactIds: ["artifact-release-note"],
        executionArtifactPaths: ["docs/release-note.md"],
        sourceArtifactId: "artifact-fix-execution-1",
        sourceArtifactPath:
          ".lime/harness/sessions/session-1/review/fix-result.json",
      },
      refs: {
        artifactIds: ["artifact-release-note"],
        artifactPaths: ["docs/release-note.md"],
      },
      rawEventRef: "artifact-fix-execution-1",
    });
  });

  it("应把 plan approval metadata 映射为结构化 action.required", () => {
    const events = buildAgentUiProjectionEvents(
      {
        type: "tool_end",
        tool_id: "tool-plan",
        result: {
          success: true,
          output: "已提交计划审批",
          metadata: {
            plan_approval_request: {
              type: "plan_approval_request",
              from: "researcher",
              requestId: "plan-req-1",
              planFilePath: "plans/alpha.md",
              planContent: "# 计划\n- 第一步",
              timestamp: "2026-05-09T00:00:00.000Z",
            },
            plan_approval_delivery: {
              target: "lead-session",
              submissionId: "submit-1",
            },
            pending_request_id: "plan-req-1",
          },
        },
      },
      baseContext,
    );

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "tool.result",
      toolCallId: "tool-plan",
    });
    expect(events[1]).toMatchObject({
      type: "action.required",
      sourceType: "tool_end",
      sequence: 11,
      actionId: "plan-req-1",
      toolCallId: "tool-plan",
      owner: "action",
      scope: "action_request",
      phase: "waiting",
      surface: "hitl",
      persistence: "snapshot",
      control: "approve",
      payload: {
        actionType: "plan_approval",
        decisionKind: "plan_approval_request",
        from: "researcher",
        planFilePath: "plans/alpha.md",
        planContentPreview: "# 计划\n- 第一步",
        planContentLength: 10,
        deliveryTarget: "lead-session",
        deliverySubmissionId: "submit-1",
        awaitingLeaderApproval: true,
      },
    });
  });

  it("应把 plan approval response source event 映射为结构化 action.resolved", () => {
    const directEvents = buildAgentUiProjectionEvents(
      {
        type: "action_resolved",
        request_id: "plan-req-1",
        action_type: "plan_approval",
        approved: false,
        feedback: "请补充验收项",
        permission_mode: "default",
        scope: {
          session_id: "child-session",
        },
        data: {
          decision_kind: "plan_approval_response",
          target_session_id: "child-session",
          plan_file: "/tmp/lime-child/PLAN.md",
          plan_id: "plan-1",
          awaiting_leader_approval: false,
        },
      },
      baseContext,
    );

    expect(directEvents).toHaveLength(1);
    expect(directEvents[0]).toMatchObject({
      type: "action.resolved",
      sourceType: "action_resolved",
      actionId: "plan-req-1",
      sessionId: "child-session",
      owner: "action",
      scope: "action_request",
      phase: "completed",
      surface: "hitl",
      persistence: "snapshot",
      control: "reject",
      payload: {
        actionType: "plan_approval",
        decisionKind: "plan_approval_response",
        approved: false,
        feedbackPreview: "请补充验收项",
        permissionMode: "default",
        targetSessionId: "child-session",
        planFile: "/tmp/lime-child/PLAN.md",
        planId: "plan-1",
        awaitingLeaderApproval: false,
      },
    });

    const toolEvents = buildAgentUiProjectionEvents(
      {
        type: "tool_end",
        tool_id: "tool-send-message",
        result: {
          success: true,
          output: "结构化发送结果",
          metadata: {
            send_message: {
              target: "researcher",
              plan_approval_response: {
                type: "plan_approval_response",
                request_id: "plan-req-2",
                approved: true,
                delivery_submission_id: "submit-response-1",
                target_session_id: "child-session",
              },
            },
          },
        },
      },
      baseContext,
    );

    expect(toolEvents).toHaveLength(2);
    expect(toolEvents[1]).toMatchObject({
      type: "action.resolved",
      sourceType: "tool_end",
      actionId: "plan-req-2",
      toolCallId: "tool-send-message",
      control: "approve",
      payload: {
        actionType: "plan_approval",
        decisionKind: "plan_approval_response",
        approved: true,
        deliveryTarget: "researcher",
        deliverySubmissionId: "submit-response-1",
        targetSessionId: "child-session",
      },
    });
  });

  it("应把 turn context、task profile 与 routing/cost 事件映射到标准 taxonomy", () => {
    const turnContextEvents = buildAgentUiProjectionEvents(
      {
        type: "turn_context",
        session_id: "session-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        output_schema_runtime: {
          source: "turn",
          strategy: "native",
          providerName: "openai",
          modelName: "gpt-5.4",
        },
        approval_policy: "on-request",
        sandbox_policy: "workspace-write",
        context_summary: {
          memory_budget: {
            used_tokens: 640,
            max_tokens: 1200,
            status: "ready",
            source: "knowledge_context_resolver",
          },
          missing_context: [
            {
              id: "knowledge_warning:0",
              kind: "knowledge_warning",
              label: "sources/missing.md",
              status: "unknown",
              reason: "缺少来源",
              source: "knowledge_context_resolver",
            },
          ],
          retrieval_refs: [
            {
              source_id: "knowledge_pack:brand:compiled/splits/brief.md",
              kind: "knowledge_pack",
              title: "brand:brief",
              path: "compiled/splits/brief.md",
              scope: "workspace",
              status: "ready",
              source: "knowledge_context_resolver",
            },
          ],
          team_memory_refs: [
            {
              key: "team.selection",
              repo_scope: "/repo/lime",
              updated_at: 1710000000,
              source: "context",
            },
          ],
        },
      },
      baseContext,
    );

    expect(turnContextEvents[0]).toMatchObject({
      type: "context.changed",
      sourceType: "turn_context",
      sessionId: "session-1",
      threadId: "thread-1",
      turnId: "turn-1",
      owner: "context",
      scope: "turn",
      phase: "preparing",
      surface: "runtime_status",
      payload: {
        outputSchemaAvailable: true,
        outputSchemaSource: "turn",
        outputSchemaStrategy: "native",
        providerName: "openai",
        modelName: "gpt-5.4",
        contextSummaryAvailable: true,
        memoryBudget: {
          used_tokens: 640,
          max_tokens: 1200,
          status: "ready",
          source: "knowledge_context_resolver",
        },
        retrievalRefs: [
          expect.objectContaining({
            source_id: "knowledge_pack:brand:compiled/splits/brief.md",
          }),
        ],
        teamMemoryRefs: [
          expect.objectContaining({
            key: "team.selection",
          }),
        ],
      },
      refs: {
        contextSourceIds: ["knowledge_pack:brand:compiled/splits/brief.md"],
        teamMemoryKeys: ["team.selection"],
      },
    });
    expect(turnContextEvents[1]).toMatchObject({
      type: "permission.changed",
      sourceType: "turn_context",
      owner: "policy",
      scope: "turn",
      phase: "preparing",
      payload: {
        approvalPolicy: "on-request",
        sandboxPolicy: "workspace-write",
        sourceEvent: "turn_context",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "task_profile_resolved",
          task_profile: {
            kind: "research",
            source: "runtime",
            traits: ["web"],
            routingSlot: "deep-search",
            permissionProfileKeys: ["read_files"],
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "task.changed",
      sourceType: "task_profile_resolved",
      owner: "task",
      scope: "run",
      phase: "routing",
      surface: "task_capsule",
      payload: {
        kind: "research",
        source: "runtime",
        traits: ["web"],
        routingSlot: "deep-search",
        permissionProfileKeys: ["read_files"],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "routing_decision_made",
          routing_decision: {
            routingMode: "auto",
            decisionSource: "runtime",
            decisionReason: "capability_match",
            selectedProvider: "openai",
            selectedModel: "gpt-5.4",
            candidateCount: 2,
            fallbackChain: ["gpt-5.4"],
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      sourceType: "routing_decision_made",
      phase: "routing",
      payload: {
        runtimeEvent: "routing_decision_made",
        routingMode: "auto",
        decisionSource: "runtime",
        selectedProvider: "openai",
        selectedModel: "gpt-5.4",
        candidateCount: 2,
        fallbackChain: ["gpt-5.4"],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "routing_fallback_applied",
          routing_decision: {
            routingMode: "profile_slot",
            decisionSource: "profile_model_slot",
            selectedProvider: "openai",
            selectedModel: "gpt-4.1-mini",
            fallbackApplied: true,
            requestedSelection: {
              provider: "custom-coding",
              model: "coder-large",
            },
            routingAttempts: [
              {
                slot: "coding",
                provider: "custom-coding",
                model: "coder-large",
                providerReadiness: {
                  status: "needs_setup",
                  reasonCode: "missing_enabled_api_key",
                },
              },
            ],
          },
        } as unknown as AgentEvent,
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      sourceType: "routing_fallback_applied",
      phase: "routing",
      payload: {
        runtimeEvent: "routing_fallback_applied",
        routingMode: "profile_slot",
        decisionSource: "profile_model_slot",
        selectedProvider: "openai",
        selectedModel: "gpt-4.1-mini",
        fallbackApplied: true,
        requestedSelection: {
          provider: "custom-coding",
          model: "coder-large",
        },
        routingAttempts: [
          {
            slot: "coding",
            provider: "custom-coding",
            model: "coder-large",
            providerReadiness: {
              status: "needs_setup",
              reasonCode: "missing_enabled_api_key",
            },
          },
        ],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "routing_decision_made",
          routing_decision: {
            routing_mode: "single_candidate",
            decision_source: "runtime",
            decision_reason: "single_configured_model",
            selected_provider: "deepseek",
            selected_model: "deepseek-chat",
            candidate_count: 1,
            fallback_chain: ["deepseek-chat"],
          },
        } as unknown as AgentEvent,
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      sourceType: "routing_decision_made",
      phase: "routing",
      payload: {
        runtimeEvent: "routing_decision_made",
        routingMode: "single_candidate",
        decisionSource: "runtime",
        decisionReason: "single_configured_model",
        selectedProvider: "deepseek",
        selectedModel: "deepseek-chat",
        candidateCount: 1,
        fallbackChain: ["deepseek-chat"],
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "routing_decision_made",
        } as unknown as AgentEvent,
        baseContext,
      )[0],
    ).toMatchObject({
      type: "run.status",
      sourceType: "routing_decision_made",
      phase: "routing",
      payload: {
        runtimeEvent: "routing_decision_made",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "cost_recorded",
          cost_state: {
            status: "estimated",
            estimatedCostClass: "low",
            estimatedTotalCost: 0.01,
            currency: "USD",
            totalTokens: 1200,
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "metric.changed",
      sourceType: "cost_recorded",
      owner: "diagnostics",
      payload: {
        metricEvent: "cost_recorded",
        status: "estimated",
        estimatedCostClass: "low",
        estimatedTotalCost: 0.01,
        currency: "USD",
        totalTokens: 1200,
      },
    });
  });

  it("应把 timeline thread item 投影为标准 process / artifact / action 事实", () => {
    const startedAt = "2026-05-09T00:00:00Z";
    const updatedAt = "2026-05-09T00:00:01Z";
    const common = {
      thread_id: "thread-1",
      turn_id: "turn-1",
      sequence: 1,
      started_at: startedAt,
      created_at: startedAt,
      updated_at: updatedAt,
    };

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_completed",
          item: {
            ...common,
            id: "reasoning-1",
            type: "reasoning",
            status: "completed",
            text: "完整推理",
            summary: ["完成推理"],
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "reasoning.summary",
      threadId: "thread-1",
      turnId: "turn-1",
      partId: "reasoning-1",
      phase: "completed",
      persistence: "archive",
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_updated",
          item: {
            ...common,
            id: "input-1",
            type: "request_user_input",
            status: "in_progress",
            request_id: "ask-1",
            action_type: "ask_user",
            prompt: "请选择",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "action.required",
      actionId: "ask-1",
      control: "answer",
      surface: "hitl",
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_completed",
          item: {
            ...common,
            id: "command-1",
            type: "command_execution",
            status: "completed",
            command: "npm test",
            cwd: "/tmp/project",
            aggregated_output: "通过",
            exit_code: 0,
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.result",
      toolCallId: "command-1",
      owner: "tool",
      scope: "tool_call",
      phase: "completed",
      payload: {
        toolName: "command_execution",
        commandPreview: "npm test",
        cwd: "/tmp/project",
        exitCode: 0,
        outputPreview: "通过",
      },
    });

    const planToolEvents = buildAgentUiProjectionEvents(
      {
        type: "item_completed",
        item: {
          ...common,
          id: "tool-plan",
          type: "tool_call",
          status: "completed",
          tool_name: "request_user_input",
          success: true,
          output: "已提交计划审批",
          metadata: {
            plan_approval_request: {
              type: "plan_approval_request",
              from: "researcher",
              request_id: "plan-req-2",
              plan_file_path: "plans/beta.md",
              plan_content: "# Beta",
            },
          },
        },
      },
      baseContext,
    );

    expect(planToolEvents).toHaveLength(2);
    expect(planToolEvents[1]).toMatchObject({
      type: "action.required",
      sourceType: "item_completed",
      actionId: "plan-req-2",
      partId: "tool-plan",
      toolCallId: "tool-plan",
      persistence: "archive",
      payload: {
        actionType: "plan_approval",
        planFilePath: "plans/beta.md",
      },
    });

    const taskUpdateEvents = buildAgentUiProjectionEvents(
      {
        type: "item_completed",
        item: {
          ...common,
          id: "tool-task-update",
          type: "tool_call",
          status: "completed",
          tool_name: "TaskUpdate",
          success: true,
          output: '{"success":true,"taskId":"1"}',
          metadata: {
            task_id: "1",
            task_list_id: "board-main",
            updated_fields: ["owner"],
            owner_change: {
              from: "researcher",
              to: "implementer",
            },
            task: {
              id: "1",
              owner: "implementer",
            },
          },
        },
      },
      baseContext,
    );

    expect(taskUpdateEvents).toHaveLength(2);
    expect(taskUpdateEvents[1]).toMatchObject({
      type: "task.changed",
      sourceType: "item_completed",
      taskId: "1",
      workItemId: "1",
      toolCallId: "tool-task-update",
      control: "assign",
      phase: "accepted",
      surface: "work_board",
      runtimeEntity: "work_item",
      runtimeStatus: "accepted",
      payload: {
        taskEvent: "team_reassignment",
        action: "reassign",
        previousAssigneeId: "researcher",
        nextAssigneeId: "implementer",
        sourceToolName: "TaskUpdate",
        sourceTaskListId: "board-main",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_completed",
          item: {
            ...common,
            id: "search-1",
            type: "web_search",
            status: "completed",
            query: "Agent UI taxonomy",
            action: "search",
            output: "找到 3 条结果",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "tool.result",
      toolCallId: "search-1",
      payload: {
        toolName: "web_search",
        queryPreview: "Agent UI taxonomy",
        action: "search",
        outputPreview: "找到 3 条结果",
      },
    });

    expect(
      buildAgentUiProjectionEvents(
        {
          type: "item_completed",
          item: {
            ...common,
            id: "summary-1",
            type: "turn_summary",
            status: "completed",
            text: "本轮已完成检索与归档。",
          },
        },
        baseContext,
      )[0],
    ).toMatchObject({
      type: "state.snapshot",
      owner: "session",
      scope: "turn",
      phase: "archived",
      surface: "timeline_evidence",
      persistence: "archive",
      payload: {
        textLength: 11,
        preview: "本轮已完成检索与归档。",
      },
    });
  });

  it("应提供 evidence.changed adapter helper，供 evidence pack 主链接入", () => {
    expect(
      buildAgentUiEvidenceChangedEvent(
        {
          evidenceId: "evidence-1",
          runId: "run-1",
          kind: "evidence_pack",
          status: "ready",
          verdict: "pass",
          summaryPreview: "已导出 evidence pack",
          artifactIds: ["artifact-1", "artifact-1"],
          artifactPaths: [".lime/evidence/pack.json"],
          itemCount: 3,
        },
        baseContext,
      ),
    ).toMatchObject({
      type: "evidence.changed",
      sourceType: "evidence_projection",
      sequence: 10,
      evidenceId: "evidence-1",
      owner: "evidence",
      scope: "evidence",
      phase: "completed",
      surface: "timeline_evidence",
      persistence: "evidence_pack",
      refs: {
        artifactIds: ["artifact-1"],
        artifactPaths: [".lime/evidence/pack.json"],
      },
      payload: {
        kind: "evidence_pack",
        status: "ready",
        verdict: "pass",
        summaryPreview: "已导出 evidence pack",
        itemCount: 3,
      },
    });
  });

  it("应提供 handoff helper，把 analysis handoff 投影到 handoff lane", () => {
    const events = buildAgentUiHandoffProjectionEvents(
      {
        evidenceId: "analysis/root",
        handoffId: "handoff/root",
        sessionId: "session-1",
        threadId: "thread-1",
        kind: "analysis_handoff",
        status: "handoff_requested",
        verdict: "complete",
        from: "lime_harness",
        to: "external_reviewer",
        reason: "analysis_handoff_exported",
        resumeTarget: "analysis/root",
        contextBoundary: "/workspace/sanitized",
        summaryPreview: "已导出外部分析文件",
        artifactPaths: ["analysis/brief.md"],
        itemCount: 1,
      },
      baseContext,
    );

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({
      type: "evidence.changed",
      evidenceId: "analysis/root",
    });
    expect(events[1]).toMatchObject({
      type: "agent.handoff",
      sourceType: "evidence_projection",
      evidenceId: "analysis/root",
      handoffId: "handoff/root",
      owner: "agent",
      scope: "agent",
      phase: "waiting",
      surface: "handoff_lane",
      persistence: "evidence_pack",
      topology: "specialist_handoff",
      payload: {
        handoffEvent: "analysis_handoff",
        status: "handoff_requested",
        verdict: "complete",
        from: "lime_harness",
        to: "external_reviewer",
        reason: "analysis_handoff_exported",
        resumeTarget: "analysis/root",
        contextBoundary: "/workspace/sanitized",
        summaryPreview: "已导出外部分析文件",
        itemCount: 1,
      },
      refs: {
        artifactPaths: ["analysis/brief.md"],
      },
    });
  });

  it("应按真实 handoff 状态映射 accepted / returned / resumed phase", () => {
    const cases = [
      ["accepted", "accepted"],
      ["returned", "reconciling"],
      ["resumed", "completed"],
      ["handoff_requested", "waiting"],
    ] as const;

    for (const [status, phase] of cases) {
      const events = buildAgentUiHandoffProjectionEvents(
        {
          evidenceId: `handoff/${status}`,
          handoffId: `handoff/${status}`,
          kind: "specialist_handoff",
          status,
          from: "coordinator",
          to: "specialist",
          resumeTarget: "agent-runtime://session/specialist",
          contextBoundary: "workspace_root",
        },
        baseContext,
      );

      expect(events[1]).toMatchObject({
        type: "agent.handoff",
        phase,
        surface: "handoff_lane",
        topology: "specialist_handoff",
        payload: {
          status,
          handoffEvent: "specialist_handoff",
          from: "coordinator",
          to: "specialist",
        },
      });
    }
  });

  it("应提供 review helper，把 review request / completed 投影到 review lane", () => {
    const requested = buildAgentUiReviewProjectionEvents(
      {
        reviewEvent: "requested",
        evidenceId: "review/root",
        reviewId: "review/root",
        sessionId: "session-1",
        threadId: "thread-1",
        kind: "review_decision",
        status: "ready",
        verdict: "pending_review",
        decisionStatus: "pending_review",
        riskLevel: "unknown",
        checklistCount: 3,
        requestedFixes: ["核对 evidence/runtime.json"],
        regressionRequirements: ["重新运行 npm run test:contracts"],
        summaryPreview: "已导出人工审核文件",
        artifactPaths: ["review/review-decision.md"],
        itemCount: 2,
      },
      baseContext,
    );

    expect(requested).toHaveLength(2);
    expect(requested[1]).toMatchObject({
      type: "review.requested",
      evidenceId: "review/root",
      reviewId: "review/root",
      owner: "evidence",
      scope: "evidence",
      phase: "reviewing",
      surface: "review_lane",
      persistence: "evidence_pack",
      control: "request_review",
      topology: "review_team",
      payload: {
        reviewEvent: "requested",
        decisionStatus: "pending_review",
        riskLevel: "unknown",
        checklistCount: 3,
        requestedFixes: ["核对 evidence/runtime.json"],
        regressionRequirements: ["重新运行 npm run test:contracts"],
      },
      refs: {
        artifactPaths: ["review/review-decision.md"],
      },
    });

    const completed = buildAgentUiReviewProjectionEvents(
      {
        reviewEvent: "completed",
        evidenceId: "review/root",
        reviewId: "review/root",
        sessionId: "session-1",
        kind: "review_decision",
        status: "completed",
        verdict: "accepted",
        decisionStatus: "accepted",
        reviewer: "human",
        riskLevel: "low",
        followupActionCount: 1,
        regressionRequirementCount: 2,
        followupActions: ["补一条 release note"],
        regressionRecoveredOutcomes: ["Artifact 校验已恢复 1 个产物。"],
        regressionRequirements: ["npm run test:contracts", "人工审核回归"],
        requestedFixExecutionResults: [
          {
            requestedFixIndex: 1,
            requestedFix: "补一条 release note",
            executionStatus: "completed",
            regressionOutcome: "recovered",
            summaryPreview: "release note 已补齐并完成回归。",
            resultRef:
              "agent-runtime://session/session-1/thread/thread-1/turn/turn-review/item/item-fix-1",
            artifactIds: ["artifact-fix-1"],
            artifactPaths: ["docs/release-note.md"],
          },
        ],
        summaryPreview: "已接受",
      },
      baseContext,
    );

    expect(completed).toHaveLength(4);
    expect(completed[1]).toMatchObject({
      type: "review.completed",
      phase: "completed",
      surface: "review_lane",
      control: "open_detail",
      payload: {
        reviewEvent: "completed",
        verdict: "accepted",
        decisionStatus: "accepted",
        reviewer: "human",
        riskLevel: "low",
        followupActionCount: 1,
        regressionRequirementCount: 2,
        regressionOutcome: "recovered",
        regressionRecoveredOutcomes: ["Artifact 校验已恢复 1 个产物。"],
        requestedFixes: ["补一条 release note"],
        followupActions: ["补一条 release note"],
        regressionRequirements: ["npm run test:contracts", "人工审核回归"],
      },
    });
    expect(completed[2]).toMatchObject({
      type: "agent.changed",
      sourceType: "evidence_projection",
      evidenceId: "review/root",
      reviewId: "review/root",
      workItemId: "review/root",
      agentId: "review/root:reviewer:human",
      agentName: "human",
      agentRole: "reviewer",
      owner: "agent",
      scope: "agent",
      phase: "completed",
      surface: "team_roster",
      persistence: "snapshot",
      control: "open_detail",
      topology: "review_team",
      runtimeEntity: "work_item",
      runtimeStatus: "completed",
      payload: {
        agentEvent: "reviewer_teammate",
        reviewEvent: "completed",
        reviewId: "review/root",
        reviewer: "human",
        decisionStatus: "accepted",
        riskLevel: "low",
      },
    });
    expect(completed[3]).toMatchObject({
      type: "task.changed",
      sourceType: "evidence_projection",
      taskId: "review/root:requested-fix:1",
      workItemId: "review/root:requested-fix:1",
      reviewId: "review/root",
      owner: "task",
      scope: "task",
      phase: "completed",
      surface: "work_board",
      persistence: "snapshot",
      control: "open_detail",
      topology: "review_team",
      runtimeEntity: "work_item",
      runtimeStatus: "completed",
      payload: {
        taskEvent: "review_requested_fix",
        requestedFix: "补一条 release note",
        requestedFixIndex: 1,
        requestedFixCount: 1,
        executionStatus: "completed",
        regressionOutcome: "recovered",
        regressionRecoveredOutcomes: ["Artifact 校验已恢复 1 个产物。"],
        regressionRequirements: ["npm run test:contracts", "人工审核回归"],
        executionSummaryPreview: "release note 已补齐并完成回归。",
        executionResultRef:
          "agent-runtime://session/session-1/thread/thread-1/turn/turn-review/item/item-fix-1",
        executionArtifactIds: ["artifact-fix-1"],
        executionArtifactPaths: ["docs/release-note.md"],
      },
      refs: {
        artifactIds: ["artifact-fix-1"],
        artifactPaths: ["docs/release-note.md"],
      },
    });
  });

  it("应提供 metric.changed adapter helper，供 diagnostics 主链接入", () => {
    expect(
      buildAgentUiMetricChangedEvent({
        phase: "agentStream.firstTextDelta",
        at: 12.5,
        wallTime: Date.parse("2026-05-09T00:00:01.000Z"),
        sessionId: "session-1",
        workspaceId: "workspace-1",
        source: "home-input",
        requestId: "request-1",
        actualSessionId: "session-runtime-1",
        metrics: {
          deltaLength: 8,
        },
      }),
    ).toMatchObject({
      type: "metric.changed",
      sourceType: "performance_metric",
      timestamp: "2026-05-09T00:00:01.000Z",
      sessionId: "session-1",
      owner: "diagnostics",
      scope: "session",
      phase: "acting",
      surface: "diagnostics",
      persistence: "diagnostics_log",
      payload: {
        metricPhase: "agentStream.firstTextDelta",
        workspaceId: "workspace-1",
        source: "home-input",
        requestId: "request-1",
        actualSessionId: "session-runtime-1",
        metrics: {
          deltaLength: 8,
        },
      },
    });
  });
});
