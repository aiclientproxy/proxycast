import { describe, expect, it } from "vitest";
import { resolveWorkspaceSceneSessionProjection } from "./workspaceSceneSessionProjection";

function baseInput(
  overrides: Partial<
    Parameters<typeof resolveWorkspaceSceneSessionProjection>[0]
  > = {},
): Parameters<typeof resolveWorkspaceSceneSessionProjection>[0] {
  return {
    shouldHideCurrentSessionContent: false,
    displayMessages: ["message-1"],
    homePendingPreviewMessages: ["preview-1"],
    turns: ["turn-1"],
    effectiveThreadItems: ["thread-item-1"],
    currentTurnId: "turn-1",
    threadRead: { id: "thread-read-1" },
    executionRuntime: { id: "runtime-1" },
    planComposerPendingActions: ["pending-action-1"],
    submittedActionsInFlight: ["submitted-action-1"],
    isPreparingSend: false,
    isTaskCenterDraftSendPending: false,
    isSending: true,
    ...overrides,
  };
}

describe("resolveWorkspaceSceneSessionProjection", () => {
  it("任务中心首页遮罩当前会话且没有 pending preview 时应清空场景投影", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        shouldHideCurrentSessionContent: true,
        homePendingPreviewMessages: [],
        isPreparingSend: true,
        isTaskCenterDraftSendPending: true,
      }),
    );

    expect(projection).toEqual({
      sceneDisplayMessages: [],
      sceneTurns: [],
      sceneThreadItems: [],
      sceneCurrentTurnId: null,
      sceneThreadRead: null,
      sceneExecutionRuntime: null,
      scenePendingActions: [],
      sceneSubmittedActionsInFlight: [],
      sceneIsPreparingSend: false,
      sceneIsSending: false,
    });
  });

  it("任务中心首页遮罩当前会话时仍应优先展示 pending preview", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        shouldHideCurrentSessionContent: true,
        isTaskCenterDraftSendPending: true,
        isSending: false,
      }),
    );

    expect(projection).toEqual({
      sceneDisplayMessages: ["preview-1"],
      sceneTurns: [],
      sceneThreadItems: [],
      sceneCurrentTurnId: null,
      sceneThreadRead: null,
      sceneExecutionRuntime: null,
      scenePendingActions: [],
      sceneSubmittedActionsInFlight: [],
      sceneIsPreparingSend: true,
      sceneIsSending: false,
    });
  });

  it("当前会话无消息时应使用首页 pending preview 消息", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        displayMessages: [],
        isTaskCenterDraftSendPending: true,
        isSending: false,
      }),
    );

    expect(projection.sceneDisplayMessages).toEqual(["preview-1"]);
    expect(projection.sceneTurns).toEqual(["turn-1"]);
    expect(projection.scenePendingActions).toEqual(["pending-action-1"]);
    expect(projection.sceneIsPreparingSend).toBe(true);
    expect(projection.sceneIsSending).toBe(false);
  });

  it("当前 read model 仍在运行时应保持场景发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          active_turn_id: "turn-running",
          turns: [{ turn_id: "turn-running", status: "running" }],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(true);
  });

  it("当前 read model 的近期 running turn 应保持场景发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          active_turn_id: "turn-running",
          pending_requests: [],
          turns: [
            {
              turn_id: "turn-running",
              status: "running",
              started_at: new Date().toISOString(),
            },
          ],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(true);
  });

  it("只有后续 turn 排队时不应把已完成会话保持为发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          active_turn_id: null,
          pending_requests: [],
          turns: [
            {
              turn_id: "turn-completed",
              status: "completed",
            },
            {
              turn_id: "turn-queued",
              status: "queued",
            },
          ],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("意外退出遗留的陈旧 running turn 不应永久锁住输入框", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          active_turn_id: "turn-orphaned",
          pending_requests: [],
          turns: [
            {
              turn_id: "turn-orphaned",
              status: "running",
              started_at: "2026-03-29T00:00:00.000Z",
            },
          ],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("缺少 active_turn_id 时不应从兼容 running turn 推断场景发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "idle",
          turns: [
            { turn_id: "turn-completed", status: "completed" },
            { turn_id: "turn-running", status: "running" },
          ],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("陈旧 running 状态指向已完成 active turn 时不应永久锁住输入框", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          profile_status: "running",
          active_turn_id: "turn-stale",
          turns: [{ turn_id: "turn-stale", status: "completed" }],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("陈旧 active_turn_id 找不到对应 running turn 时应 fail closed", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          active_turn_id: "turn-missing",
          turns: [{ turn_id: "turn-completed", status: "completed" }],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("只有 thread 级 running 但没有 running turn 证据时不应进入发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "running",
          profile_status: "running",
          turns: [{ turn_id: "turn-completed", status: "completed" }],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });

  it("当前 read model 已失败时不应被残留 active turn 维持发送态", () => {
    const projection = resolveWorkspaceSceneSessionProjection(
      baseInput({
        isSending: false,
        threadRead: {
          status: "failed",
          profile_status: "failed",
          active_turn_id: "turn-stale",
          turns: [{ turn_id: "turn-stale", status: "running" }],
        },
      }),
    );

    expect(projection.sceneIsSending).toBe(false);
  });
});
