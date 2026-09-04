import { describe, expect, it, vi } from "vitest";
import {
  createBaseParams,
  getRenderedSceneProps,
} from "./useWorkspaceConversationSceneRuntime.testFixtures";

describe("useWorkspaceConversationSceneRuntime task rail projection", () => {
  it("应把运行摘要轻量事实透传给 Task Center 任务轨道", () => {
    const params = createBaseParams({
      providerType: "cloud",
      model: "reasoner-pro",
      accessMode: "current",
      reasoningEffort: "medium",
      projectRootPath: "/tmp/project-1",
      canvasWorkbenchRootPath: "/tmp/canvas-root",
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.taskRail?.providerType).toBe("cloud");
    expect(sceneProps.taskRail?.model).toBe("reasoner-pro");
    expect(sceneProps.taskRail?.accessMode).toBe("current");
    expect(sceneProps.taskRail?.reasoningEffort).toBe("medium");
    expect(sceneProps.taskRail?.workspaceRootPath).toBe("/tmp/project-1");
    expect(sceneProps.taskRail?.context).toBeUndefined();
  });

  it("应把待确认状态与响应入口透传给 Task Center 任务轨道", () => {
    const handlePermissionResponse = vi.fn();
    const pendingActions = [
      {
        requestId: "approval-write",
        actionType: "tool_confirmation",
        toolName: "write_file",
        prompt: "允许保存 result.md？",
      },
    ];
    const submittedActionsInFlight = [
      {
        requestId: "approval-shell",
        actionType: "tool_confirmation",
        toolName: "shell",
        status: "submitted",
      },
    ];
    const params = createBaseParams({
      pendingActions,
      submittedActionsInFlight,
      handlePermissionResponse,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.taskRail?.pendingActions).toBe(pendingActions);
    expect(sceneProps.taskRail?.submittedActionsInFlight).toBe(
      submittedActionsInFlight,
    );
    expect(sceneProps.taskRail?.onRespondToAction).toBe(
      handlePermissionResponse,
    );
  });

  it("应把已投影 timeline 透传给 Task Center 任务轨道用于已处理确认回显", () => {
    const threadItems = [
      {
        id: "approval-write-item",
        type: "approval_request",
        thread_id: "thread-1",
        turn_id: "turn-1",
        sequence: 1,
        status: "completed",
        request_id: "approval-write",
        action_type: "tool_confirmation",
        prompt: "允许保存 result.md？",
        response: "approved",
        started_at: "2026-06-16T10:00:00.000Z",
        completed_at: "2026-06-16T10:00:03.000Z",
        updated_at: "2026-06-16T10:00:03.000Z",
      },
    ];
    const params = createBaseParams({
      effectiveThreadItems: threadItems,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.taskRail?.threadItems).toBe(threadItems);
  });

  it("应把 todo items 透传给 Task Center 任务轨道用于恢复历史计划", () => {
    const todoItems = [
      {
        content: "恢复历史计划",
        status: "in_progress",
      },
    ];
    const params = createBaseParams({
      todoItems,
    });

    const sceneProps = getRenderedSceneProps(params);
    expect(sceneProps.taskRail?.todoItems).toBe(todoItems);
  });

  it("应把 canonical Goal、read model 变更和子任务事实透传给 Task Center 任务轨道", () => {
    const threadRead = {
      thread_id: "thread-1",
      change_summary: {
        changed_file_count: 2,
        changed_files: ["src/App.tsx", "src/index.ts"],
        patch_count: 2,
        applied_patch_count: 1,
      },
      context_summary: {
        sources: ["https://docs.example.com/task-rail"],
        retrieval_refs: [
          {
            source_id: "retrieval-1",
            kind: "file",
            title: "run-observability.md",
          },
        ],
      },
      evidence_summary: {
        evidence_refs: ["evidence/run-control.json"],
      },
    };
    const canonicalChildren = [
      {
        name: "实现",
        parentThreadId: "thread-parent",
        sessionId: "child-running",
        status: "running",
        threadId: "thread-child-running",
        updatedAtMs: 2,
      },
      {
        name: "验证",
        parentThreadId: "thread-parent",
        sessionId: "child-done",
        status: "completed",
        threadId: "thread-child-done",
        updatedAtMs: 2,
      },
    ];
    const params = createBaseParams({
      threadGoal: {
        createdAt: 1,
        objective: "完成任务区域摘要",
        status: "active",
        threadId: "thread-1",
        timeUsedSeconds: 0,
        tokensUsed: 0,
        updatedAt: 1,
      },
      threadRead,
      canonicalChildren,
    });

    const sceneProps = getRenderedSceneProps(params);

    expect(sceneProps.taskRail?.threadRead).toBe(threadRead);
    expect(sceneProps.taskRail?.threadGoal?.objective).toBe(
      "完成任务区域摘要",
    );
    expect(sceneProps.taskRail?.canonicalChildren).toBe(canonicalChildren);
    expect(sceneProps.taskRail?.context).toBeUndefined();
  });

  it("需要用户关注时才向画布壳透传会话进展面板", () => {
    const handlePermissionResponse = vi.fn();
    const params = createBaseParams({
      handlePermissionResponse,
      turns: [
        {
          id: "turn-1",
          thread_id: "thread-1",
          prompt_text: "请抓取文章并整理成 markdown",
          status: "running",
          started_at: "2026-04-09T10:00:00.000Z",
          created_at: "2026-04-09T10:00:00.000Z",
          updated_at: "2026-04-09T10:00:01.000Z",
        },
      ],
      currentTurnId: "turn-1",
      effectiveThreadItems: [
        {
          id: "item-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 1,
          status: "in_progress",
          started_at: "2026-04-09T10:00:00.000Z",
          updated_at: "2026-04-09T10:00:01.000Z",
          type: "command_execution",
          command: "curl https://example.com/article",
          cwd: "/tmp/project-1",
        },
        {
          id: "action-1",
          thread_id: "thread-1",
          turn_id: "turn-1",
          sequence: 2,
          status: "in_progress",
          started_at: "2026-04-09T10:00:01.000Z",
          updated_at: "2026-04-09T10:00:02.000Z",
          type: "request_user_input",
          request_id: "req-1",
          action_type: "elicitation",
          prompt: "请补充导出目录",
        },
      ],
      pendingActions: [
        {
          requestId: "req-1",
          actionType: "elicitation",
          prompt: "请补充导出目录",
          status: "pending",
        },
      ],
      settledWorkbenchArtifacts: [{ id: "artifact-1" }],
      isSending: true,
      focusedTimelineItemId: "item-1",
    });

    const sceneProps = getRenderedSceneProps(params);
    const sessionView = sceneProps.canvasWorkbenchLayoutProps.sessionView;

    expect(sessionView?.title).toBe("任务进展");
    expect(sessionView?.tabLabel).toBe("进展");
    expect(sessionView?.tabBadge).toBe("处理中 1");
    expect(sessionView?.tabBadgeTone).toBe("sky");
    expect(sessionView?.subtitle).toBe("正在处理：请抓取文章并整理成 markdown");
    expect(sessionView?.summaryStats).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "session-status",
          label: "当前状态",
          value: "执行中",
        }),
        expect.objectContaining({
          key: "session-generated-files",
          label: "生成内容",
          value: "处理中 1",
        }),
        expect.objectContaining({
          key: "session-follow-up",
          label: "需要你处理",
          value: "待补充 1",
        }),
      ]),
    );
    expect(sessionView?.badges).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          key: "session-status",
          label: "执行中",
        }),
        expect.objectContaining({
          key: "session-pending-actions",
          label: "待补充 1",
        }),
      ]),
    );
    expect(typeof sessionView?.renderPanel).toBe("function");
  });

  it("任务中心输出文件应按工作区根目录解析后再打开", () => {
    const handleOpenCanvasWorkbenchPath = vi.fn();
    const sceneProps = getRenderedSceneProps(
      createBaseParams({
        canvasScene: {
          ...createBaseParams().canvasScene,
          handleOpenCanvasWorkbenchPath,
        },
        projectRootPath: "/tmp/project-1",
        canvasWorkbenchRootPath: "/tmp/session-root",
        steps: [{ id: "write", title: "整理输出文件", status: "active" }],
      }),
    );

    sceneProps.taskRail?.onOpenOutput?.("docs/result.md");
    expect(handleOpenCanvasWorkbenchPath).toHaveBeenCalledWith(
      "/tmp/project-1/docs/result.md",
    );

    sceneProps.taskRail?.onOpenOutput?.("/tmp/absolute.md");
    expect(handleOpenCanvasWorkbenchPath).toHaveBeenLastCalledWith(
      "/tmp/absolute.md",
    );
  });
});
