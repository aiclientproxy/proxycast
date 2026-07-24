import fs from "node:fs";
import { describe, expect, it } from "vitest";

import {
  APPROVAL_REQUEST_RESUME_APPROVAL_PROMPT,
  APPROVAL_REQUEST_RESUME_COMMAND,
  APPROVAL_REQUEST_RESUME_PROMPT,
  APPROVAL_REQUEST_RESUME_REQUEST_ID,
  APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
} from "./claw-chat-current-fixture-constants.mjs";
import { summarizeApprovalPendingReadModel } from "./claw-chat-current-fixture-approval-read-model.mjs";

const sourcePath =
  "scripts/agent-runtime/claw-chat-current-fixture-approval-read-model.mjs";

describe("claw approval pending read model", () => {
  it("通过 thread/read 与 renderer lifecycle 读取 typed approval，不重复 resume", () => {
    const source = fs.readFileSync(sourcePath, "utf8");

    expect(source).toContain("APP_SERVER_METHOD_SESSION_READ");
    expect(source).toContain(
      "lime:debug:app-server-server-request-lifecycle:v1",
    );
    expect(source).not.toContain("APP_SERVER_METHOD_SESSION_THREAD_RESUME");
    expect(source).not.toContain("resume.messages");
    expect(source).not.toContain("agentSession/action/replay");
  });

  it("从 thread/read canonical item 与脱敏 lifecycle 汇总审批事实", () => {
    const readModel = {
      thread: {
        turns: [
          {
            id: "turn-1",
            status: "inProgress",
            items: [
              { content: APPROVAL_REQUEST_RESUME_PROMPT },
              {
                id: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
                type: "commandExecution",
                command: APPROVAL_REQUEST_RESUME_COMMAND,
              },
            ],
          },
        ],
      },
    };
    const summary = summarizeApprovalPendingReadModel(readModel, [
      {
        kind: "request",
        id: "outer-approval-1",
        method: "item/commandExecution/requestApproval",
        threadId: "thread-1",
        turnId: "turn-1",
        itemId: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
        approvalId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
      },
    ]);

    expect(summary).toMatchObject({
      hasPendingRequest: true,
      includesApprovalPrompt: false,
      includesCommand: true,
      includesPrompt: true,
      includesRequestId: true,
      includesToolCallId: true,
      itemId: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
      latestTurnStatus: "inProgress",
      outerRequestId: "outer-approval-1",
      payloadActionType: "tool_confirmation",
      payloadToolName: "exec_command",
      pendingRequestCount: 1,
      readModelPendingRequestCount: 0,
      requestId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
      requestStatus: "pending",
      requestType: "item/commandExecution/requestApproval",
      threadId: "thread-1",
      turnId: "turn-1",
    });
  });

  it("lifecycle 不携带匹配 request 时保持 fail closed", () => {
    const summary = summarizeApprovalPendingReadModel(
      {
        thread: {
          turns: [
            {
              status: "inProgress",
              items: [
                {
                  id: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
                  type: "commandExecution",
                  command: APPROVAL_REQUEST_RESUME_COMMAND,
                },
              ],
            },
          ],
        },
      },
      [
        {
          kind: "request",
          id: "unrelated",
          method: "item/commandExecution/requestApproval",
          approvalId: "another-approval",
        },
      ],
    );

    expect(summary).toMatchObject({
      hasPendingRequest: false,
      outerRequestId: null,
      pendingRequestCount: 0,
      readModelPendingRequestCount: 0,
      requestId: null,
    });
  });

  it("已响应的 lifecycle request 不再计为 pending", () => {
    const summary = summarizeApprovalPendingReadModel(
      {
        thread: {
          turns: [
            {
              status: "inProgress",
              items: [
                {
                  id: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
                  type: "commandExecution",
                  command: APPROVAL_REQUEST_RESUME_COMMAND,
                },
              ],
            },
          ],
        },
      },
      [
        {
          kind: "request",
          id: "outer-approval-1",
          method: "item/commandExecution/requestApproval",
          threadId: "thread-1",
          turnId: "turn-1",
          itemId: APPROVAL_REQUEST_RESUME_TOOL_CALL_ID,
          approvalId: APPROVAL_REQUEST_RESUME_REQUEST_ID,
        },
        {
          kind: "response",
          id: "outer-approval-1",
          decision: "acceptForSession",
        },
      ],
    );

    expect(summary).toMatchObject({
      hasPendingRequest: false,
      outerRequestId: null,
      pendingRequestCount: 0,
      requestId: null,
    });
  });
});
