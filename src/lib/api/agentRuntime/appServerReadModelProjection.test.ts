import { describe, expect, it } from "vitest";
import type { AppServerAgentSessionReadResponse } from "@/lib/api/appServer";
import {
  projectAppServerSessionReadToThreadReadModel,
  type AppServerAgentSessionReadProjectionInput,
} from "./appServerReadModelProjection";

function sessionRead(
  overrides: Partial<AppServerAgentSessionReadResponse> & {
    detail?: unknown;
  } = {},
): AppServerAgentSessionReadProjectionInput {
  return {
    session: {
      sessionId: "session-1",
      threadId: "thread-1",
      appId: "agent-chat",
      workspaceId: "workspace-1",
      status: "running",
      createdAt: "2026-06-06T00:00:00.000Z",
      updatedAt: "2026-06-06T00:00:03.000Z",
    },
    turns: [
      {
        turnId: "turn-1",
        sessionId: "session-1",
        threadId: "thread-1",
        status: "completed",
        startedAt: "2026-06-06T00:00:01.000Z",
        completedAt: "2026-06-06T00:00:02.000Z",
      },
      {
        turnId: "turn-2",
        sessionId: "session-1",
        threadId: "thread-1",
        status: "running",
        startedAt: "2026-06-06T00:00:03.000Z",
      },
    ],
    ...overrides,
  };
}

describe("appServerReadModelProjection", () => {
  it("应把 App Server session + turns 投影成前端 thread read model", () => {
    const result = projectAppServerSessionReadToThreadReadModel(sessionRead());

    expect(result).toEqual({
      thread_id: "thread-1",
      status: "running",
      profile_status: "running",
      active_turn_id: undefined,
      updated_at: "2026-06-06T00:00:03.000Z",
      pending_requests: [],
      incidents: [],
      turns: [
        {
          turn_id: "turn-1",
          status: "completed",
          native_status: "completed",
        },
        {
          turn_id: "turn-2",
          status: "running",
          native_status: "running",
        },
      ],
    });
  });

  it("没有 canonical detail 时不应从 protocol turns 推断 active turn", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        turns: [
          {
            turnId: "turn-running",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "running",
            startedAt: "2026-06-06T00:00:01.000Z",
          },
          {
            turnId: "turn-queued",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "queued",
            startedAt: "2026-06-06T00:00:02.000Z",
          },
        ],
      }),
    );

    expect(result.active_turn_id).toBeUndefined();
  });

  it("detail 明确没有 active turn 时不应从兼容 turns 反向推断", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        detail: {
          thread_read: {
            thread_id: "thread-1",
            status: "running",
            active_turn_id: null,
          },
        },
        turns: [
          {
            turnId: "turn-completed",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "completed",
            startedAt: "2026-06-06T00:00:01.000Z",
            completedAt: "2026-06-06T00:00:02.000Z",
          },
          {
            turnId: "turn-compat-running",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "running",
            startedAt: "2026-06-06T00:00:03.000Z",
          },
        ],
      }),
    );

    expect(result.active_turn_id).toBeUndefined();
  });

  it("应保留 detail.thread_read 的富 read model", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        detail: {
          thread_read: {
            thread_id: "thread-from-detail",
            status: "blocked",
            profile_status: "blocked",
            active_turn_id: "turn-detail",
            turns: [
              {
                turn_id: "turn-detail",
                status: "blocked",
                native_status: "waitingAction",
              },
            ],
            pending_requests: [
              {
                id: "request-1",
                thread_id: "thread-1",
                turn_id: "turn-detail",
                request_type: "ask_user",
                status: "pending",
              },
            ],
            active_command_id: "command-active",
            active_test_run_id: "test-active",
            active_action_id: "request-1",
            commands: [
              {
                command_id: "command-active",
                status: "running",
                command: "npm test",
                cwd: "app",
                output_refs: ["output://command-active"],
                output_preview: "running tests",
              },
            ],
            tests: [
              {
                test_run_id: "test-active",
                status: "running",
                command_id: "command-active",
                suite: "unit",
                passed: 3,
                failed: 0,
              },
            ],
          },
        },
      }),
    );

    expect(result).toMatchObject({
      thread_id: "thread-1",
      status: "blocked",
      profile_status: "blocked",
      active_turn_id: "turn-detail",
      turns: [
        {
          turn_id: "turn-detail",
          status: "blocked",
          native_status: "waitingAction",
        },
      ],
      pending_requests: [
        {
          id: "request-1",
          request_type: "ask_user",
          status: "pending",
        },
      ],
      active_command_id: "command-active",
      active_test_run_id: "test-active",
      active_action_id: "request-1",
      commands: [
        {
          command_id: "command-active",
          status: "running",
          command: "npm test",
          cwd: "app",
          output_refs: ["output://command-active"],
          output_preview: "running tests",
        },
      ],
      tests: [
        {
          test_run_id: "test-active",
          status: "running",
          command_id: "command-active",
          suite: "unit",
          passed: 3,
          failed: 0,
        },
      ],
    });
  });

  it("应保留 detail.thread_read 的 plan/reasoning thread_items metadata", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        detail: {
          thread_read: {
            thread_id: "thread-1",
            status: "completed",
            thread_items: [
              {
                id: "reasoning-1",
                thread_id: "thread-1",
                turn_id: "turn-1",
                sequence: 1,
                status: "completed",
                started_at: "2026-06-23T00:00:00.000Z",
                completed_at: "2026-06-23T00:00:01.000Z",
                updated_at: "2026-06-23T00:00:01.000Z",
                type: "reasoning",
                text: "先理解目标。",
                metadata: {
                  provider_metadata: {
                    backend: "codex",
                    signature: "thinking-signature",
                  },
                },
              },
              {
                id: "plan-1",
                thread_id: "thread-1",
                turn_id: "turn-1",
                sequence: 2,
                status: "completed",
                started_at: "2026-06-23T00:00:02.000Z",
                completed_at: "2026-06-23T00:00:03.000Z",
                updated_at: "2026-06-23T00:00:03.000Z",
                type: "plan",
                text: "- [ ] 验证历史恢复",
                metadata: {
                  revisionId: "plan:history",
                  plan: [{ step: "验证历史恢复", status: "pending" }],
                },
              },
            ],
          },
        },
      }),
    );

    expect(result.thread_items).toEqual([
      expect.objectContaining({
        id: "reasoning-1",
        type: "reasoning",
        metadata: {
          provider_metadata: {
            backend: "codex",
            signature: "thinking-signature",
          },
        },
      }),
      expect.objectContaining({
        id: "plan-1",
        type: "plan",
        metadata: {
          revisionId: "plan:history",
          plan: [{ step: "验证历史恢复", status: "pending" }],
        },
      }),
    ]);
  });

  it("应保留 detail.thread_read 的 workflow facts", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        detail: {
          thread_read: {
            thread_id: "thread-1",
            status: "completed",
            workflow_runs: [
              {
                workflow_run_id: "task-article:workflow",
                workflow_key: "content_article_workflow",
                status: "completed",
                steps: [
                  {
                    workflow_run_id: "task-article:workflow",
                    step_id: "research",
                    status: "completed",
                  },
                ],
              },
            ],
            workflow_steps: [
              {
                workflow_run_id: "task-article:workflow",
                step_id: "research",
                status: "completed",
              },
            ],
            workflowRuns: [
              {
                workflowRunId: "task-article:workflow",
                workflowKey: "content_article_workflow",
                status: "completed",
              },
            ],
            workflowSteps: [
              {
                workflowRunId: "task-article:workflow",
                stepId: "research",
                status: "completed",
              },
            ],
          },
        },
      }),
    );

    expect(result.workflow_runs).toEqual([
      expect.objectContaining({
        workflow_run_id: "task-article:workflow",
        workflow_key: "content_article_workflow",
        status: "completed",
      }),
    ]);
    expect(result.workflow_steps).toEqual([
      expect.objectContaining({
        workflow_run_id: "task-article:workflow",
        step_id: "research",
        status: "completed",
      }),
    ]);
    expect(result.workflowRuns).toEqual([
      expect.objectContaining({
        workflowRunId: "task-article:workflow",
        workflowKey: "content_article_workflow",
        status: "completed",
      }),
    ]);
    expect(result.workflowSteps).toEqual([
      expect.objectContaining({
        workflowRunId: "task-article:workflow",
        stepId: "research",
        status: "completed",
      }),
    ]);
  });

  it("应把 session business object metadata 投影到 thread read model", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        session: {
          sessionId: "session-1",
          threadId: "thread-1",
          appId: "agent-chat",
          workspaceId: "workspace-1",
          status: "running",
          createdAt: "2026-06-06T00:00:00.000Z",
          updatedAt: "2026-06-06T00:00:03.000Z",
          businessObjectRef: {
            kind: "agent.session",
            id: "agent-session:workspace-1:session-1",
            metadata: {
              title: "代码文学专家",
              expert: { expertId: "code-literature" },
              harness: {
                expert: { expert_id: "code-literature" },
              },
            },
          },
        },
      }),
    );

    expect(result.session_business_object_ref_metadata).toEqual({
      title: "代码文学专家",
      expert: { expertId: "code-literature" },
      harness: {
        expert: { expert_id: "code-literature" },
      },
    });
  });

  it("detail.thread_read 的 session metadata 应优先于 session business object metadata", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        session: {
          sessionId: "session-1",
          threadId: "thread-1",
          appId: "agent-chat",
          workspaceId: "workspace-1",
          status: "running",
          createdAt: "2026-06-06T00:00:00.000Z",
          updatedAt: "2026-06-06T00:00:03.000Z",
          businessObjectRef: {
            kind: "agent.session",
            id: "agent-session:workspace-1:session-1",
            metadata: {
              expert: { expertId: "session-expert" },
            },
          },
        },
        detail: {
          thread_read: {
            thread_id: "thread-1",
            session_business_object_ref_metadata: {
              expert: { expertId: "detail-expert" },
            },
          },
        },
      }),
    );

    expect(result.session_business_object_ref_metadata).toEqual({
      expert: { expertId: "detail-expert" },
    });
  });

  it("应把 App Server canceled 状态归一为前端 cancelled", () => {
    const result = projectAppServerSessionReadToThreadReadModel(
      sessionRead({
        session: {
          sessionId: "session-1",
          threadId: "thread-1",
          appId: "agent-chat",
          status: "canceled",
          createdAt: "2026-06-06T00:00:00.000Z",
          updatedAt: "2026-06-06T00:00:02.000Z",
        },
        turns: [
          {
            turnId: "turn-1",
            sessionId: "session-1",
            threadId: "thread-1",
            status: "canceled",
          },
        ],
      }),
    );

    expect(result.status).toBe("cancelled");
    expect(result.profile_status).toBe("cancelled");
    expect(result.turns).toEqual([
      {
        turn_id: "turn-1",
        status: "cancelled",
        native_status: "canceled",
      },
    ]);
  });
});
