import { describe, expect, it } from "vitest";
import type { ThreadResumeResponse } from "@limecloud/app-server-client";
import { projectAppServerAgentEventPayload } from "../appServerEventStream";
import { readCanonicalThreadDetail } from "../appServerCanonicalThreadProjection";
import { reduceConversationProjectionPayloads } from "./adapters";
import { createThreadResumeConversationProjection } from "./replay";

const CREATED_AT_SECONDS = 1_753_132_800;

function resumeResponse(): ThreadResumeResponse {
  return {
    approvalPolicy: null,
    approvalsReviewer: null,
    cwd: "/workspace",
    model: "gpt-5.4",
    modelProvider: "openai",
    sandbox: null,
    thread: {
      cliVersion: "0.0.0",
      createdAt: CREATED_AT_SECONDS,
      cwd: "/workspace",
      ephemeral: false,
      historyMode: "legacy",
      id: "thread-resume-projection",
      modelProvider: "openai",
      preview: "Resume projection",
      sessionId: "session-resume-projection",
      source: "appServer",
      status: { type: "idle" },
      turns: [
        {
          id: "turn-newer",
          status: "completed",
          startedAt: CREATED_AT_SECONDS + 2,
          completedAt: CREATED_AT_SECONDS + 3,
          items: [
            {
              id: "item-newer-agent",
              type: "agentMessage",
              text: "第二轮完成",
            },
          ],
        },
      ],
      updatedAt: CREATED_AT_SECONDS + 3,
    },
    initialTurnsPage: {
      data: [
        {
          id: "turn-newer",
          status: "inProgress",
          startedAt: CREATED_AT_SECONDS + 2,
          items: [],
        },
        {
          id: "turn-older",
          status: "completed",
          startedAt: CREATED_AT_SECONDS,
          completedAt: CREATED_AT_SECONDS + 1,
          items: [
            {
              id: "item-older-user",
              type: "userMessage",
              content: [{ type: "text", text: "第一轮" }],
            },
          ],
        },
      ],
    },
  };
}

describe("thread resume ConversationProjection", () => {
  it("合并真实 resume snapshot/page，并保留 identity、顺序和 terminal", () => {
    const reducer = createThreadResumeConversationProjection(resumeResponse());
    const projection = reducer?.getProjection();

    expect(projection).toMatchObject({
      thread_id: "thread-resume-projection",
      status: "idle",
    });
    expect(projection?.turns.map(({ id, status }) => ({ id, status }))).toEqual([
      { id: "turn-older", status: "completed" },
      { id: "turn-newer", status: "completed" },
    ]);
    expect(
      projection?.items.map(({ id, sequence, status, turn_id }) => ({
        id,
        sequence,
        status,
        turn_id,
      })),
    ).toEqual([
      {
        id: "item-older-user",
        sequence: 0,
        status: "completed",
        turn_id: "turn-older",
      },
      {
        id: "item-newer-agent",
        sequence: 0,
        status: "completed",
        turn_id: "turn-newer",
      },
    ]);
  });

  it("live notification、cold read 与 production resume replay 保持同一投影", () => {
    const response = resumeResponse();
    response.initialTurnsPage = undefined;
    const cold = readCanonicalThreadDetail({ thread: response.thread });
    const replay = createThreadResumeConversationProjection(response);
    const rawItem = response.thread.turns?.[0]?.items?.[0];
    if (!cold || !replay || !rawItem) {
      throw new Error("expected complete canonical projection fixture");
    }

    const notifications = [
      {
        method: "thread/started",
        params: { thread: response.thread },
      },
      {
        method: "turn/started",
        params: {
          threadId: response.thread.id,
          turn: {
            id: "turn-newer",
            status: "inProgress",
            startedAt: CREATED_AT_SECONDS + 2,
          },
        },
      },
      {
        method: "item/started",
        params: {
          threadId: response.thread.id,
          turnId: "turn-newer",
          startedAtMs: (CREATED_AT_SECONDS + 2) * 1_000,
          item: rawItem,
        },
      },
      {
        method: "item/completed",
        params: {
          threadId: response.thread.id,
          turnId: "turn-newer",
          completedAtMs: (CREATED_AT_SECONDS + 2) * 1_000,
          item: rawItem,
        },
      },
      {
        method: "turn/completed",
        params: {
          threadId: response.thread.id,
          turn: response.thread.turns?.[0],
        },
      },
    ]
      .map(projectAppServerAgentEventPayload)
      .filter(
        (payload): payload is Record<string, unknown> => payload !== null,
      );
    const live = reduceConversationProjectionPayloads(
      notifications,
      "live",
      response.thread.id,
    ).getProjection();
    const replayProjection = replay.getProjection();

    expect(live.items).toEqual(cold.items);
    expect(replayProjection.items).toEqual(cold.items);
    expect(
      [live, replayProjection].map((projection) =>
        projection.turns.map(({ id, status }) => ({ id, status })),
      ),
    ).toEqual([
      [{ id: "turn-newer", status: "completed" }],
      [{ id: "turn-newer", status: "completed" }],
    ]);
  });

  it("拒绝缺失 canonical identity 的 resume snapshot", () => {
    const response = resumeResponse();
    response.thread.id = "";

    expect(createThreadResumeConversationProjection(response)).toBeNull();
  });
});
