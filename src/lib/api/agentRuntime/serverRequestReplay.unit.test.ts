import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getDefaultPendingInteractionController,
  resetDefaultPendingInteractionControllerForTests,
} from "./pendingInteractionController";
import {
  findPendingTypedAction,
  findPendingTypedServerRequestAction,
  respondPendingTypedServerRequest,
  replayedActionViewFromPendingAction,
} from "./serverRequestReplay";

describe("typed server-request replay", () => {
  beforeEach(() => {
    resetDefaultPendingInteractionControllerForTests();
  });

  afterEach(() => vi.restoreAllMocks());

  it("仅从同一 session/thread 的 typed pending snapshot 重建 action", () => {
    const action = {
      requestId: "item-1",
      actionType: "ask_user" as const,
      prompt: "请选择执行模式",
      scope: { threadId: "thread-1", turnId: "turn-1" },
      status: "pending" as const,
    };
    expect(replayedActionViewFromPendingAction(action)).toMatchObject({
      type: "action_required",
      request_id: "item-1",
      action_type: "ask_user",
      prompt: "请选择执行模式",
      scope: { thread_id: "thread-1", turn_id: "turn-1" },
    });
    expect(findPendingTypedAction([action], "thread-1", "item-1")).toEqual(
      action,
    );
    expect(
      findPendingTypedAction([action], "other-thread", "item-1"),
    ).toBeNull();
    expect(
      findPendingTypedServerRequestAction("thread-1", "item-1"),
    ).toBeNull();
  });

  it("空 scope 不可被 replay", () => {
    expect(
      findPendingTypedServerRequestAction("session-1", "request-1"),
    ).toBeNull();
    expect(getDefaultPendingInteractionController().getSnapshot()).toEqual([]);
  });

  it("同作用域 AskUser typed pending 应由 controller settle", () => {
    const controller = getDefaultPendingInteractionController();
    vi.spyOn(controller, "getSnapshot").mockReturnValue([
      {
        id: "request_user_input:thread-1:turn-1:item-ask-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        item_id: "item-ask-1",
        kind: "request_user_input",
        status: "pending",
        payload: {
          request: {
            requestId: "item-ask-1",
            actionType: "ask_user",
            scope: { threadId: "thread-1", turnId: "turn-1" },
            status: "pending",
          },
        },
      },
    ]);
    const respond = vi
      .spyOn(controller, "respond")
      .mockReturnValue({ accepted: true });

    expect(
      respondPendingTypedServerRequest({
        session_id: "thread-1",
        request_id: "item-ask-1",
        action_type: "ask_user",
        confirmed: true,
        user_data: { mode: "auto" },
        action_scope: { thread_id: "thread-1", turn_id: "turn-1" },
      }),
    ).toBe(true);
    expect(respond).toHaveBeenCalledWith({
      confirmed: true,
      interactionId: "request_user_input:thread-1:turn-1:item-ask-1",
      kind: "request_user_input",
      response: undefined,
      userData: { mode: "auto" },
    });
  });

  it("typed pending scope 不匹配时 fail closed", () => {
    const controller = getDefaultPendingInteractionController();
    vi.spyOn(controller, "getSnapshot").mockReturnValue([
      {
        id: "approval:thread-1:turn-1:approval-1",
        thread_id: "thread-1",
        turn_id: "turn-1",
        kind: "approval",
        status: "pending",
        payload: {
          request: {
            requestId: "approval-1",
            actionType: "tool_confirmation",
            scope: { threadId: "thread-1", turnId: "turn-1" },
            status: "pending",
          },
        },
      },
    ]);
    const respond = vi.spyOn(controller, "respond");

    expect(
      respondPendingTypedServerRequest({
        session_id: "thread-1",
        request_id: "approval-1",
        action_type: "tool_confirmation",
        decision: "allow_once",
        action_scope: { thread_id: "thread-1", turn_id: "turn-other" },
      }),
    ).toBe(false);
    expect(respond).not.toHaveBeenCalled();
  });
});
