import { describe, expect, it } from "vitest";

import {
  readCanonicalThreadDetail,
  readCanonicalThreadListResponse,
} from "./appServerCanonicalThreadProjection";

const CREATED_AT_SECONDS = 1_780_704_000;

describe("appServerCanonicalThreadProjection", () => {
  it("从最新成功 update_plan 恢复 checklist，且不生成 Plan Item", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-plan-recovery",
        sessionId: "session-plan-recovery",
        status: { type: "idle" },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 5,
        turns: [
          {
            id: "turn-plan-recovery",
            status: "completed",
            items: [
              {
                id: "tool-plan-old",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: {
                  plan: [{ step: "旧步骤", status: "pending" }],
                },
              },
              {
                id: "tool-plan-failed",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "failed",
                success: false,
                arguments: {
                  plan: [{ step: "失败步骤", status: "completed" }],
                },
              },
              {
                id: "tool-plan-new",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: [
                  { name: "explanation", value: "继续执行" },
                  {
                    name: "plan",
                    value: JSON.stringify([
                      { step: "读现状", status: "completed" },
                      { step: "补主链", status: "in_progress" },
                    ]),
                  },
                ],
              },
            ],
          },
        ],
      },
    });

    expect(detail?.todo_items).toEqual([
      { content: "读现状", status: "completed" },
      { content: "补主链", status: "in_progress" },
    ]);
    expect(detail?.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "tool-plan-new",
          type: "tool_call",
          tool_name: "update_plan",
        }),
      ]),
    );
    expect(detail?.items.some((item) => item.type === "plan")).toBe(false);
  });

  it("空 plan 成功快照可以清空旧 checklist，非法快照不覆盖", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-plan-empty",
        sessionId: "session-plan-empty",
        status: { type: "idle" },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 5,
        turns: [
          {
            id: "turn-plan-empty",
            status: "completed",
            items: [
              {
                id: "tool-plan-valid",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: {
                  plan: [{ step: "保留步骤", status: "pending" }],
                },
              },
              {
                id: "tool-plan-malformed",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: [{ name: "plan", value: "not-json" }],
              },
            ],
          },
        ],
      },
    });

    expect(detail?.todo_items).toEqual([
      { content: "保留步骤", status: "pending" },
    ]);

    const cleared = readCanonicalThreadDetail({
      thread: {
        id: "thread-plan-clear",
        sessionId: "session-plan-clear",
        status: { type: "idle" },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 5,
        turns: [
          {
            id: "turn-plan-clear",
            status: "completed",
            items: [
              {
                id: "tool-plan-before-clear",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: {
                  plan: [{ step: "旧步骤", status: "pending" }],
                },
              },
              {
                id: "tool-plan-clear",
                type: "dynamicToolCall",
                tool: "update_plan",
                status: "completed",
                success: true,
                arguments: { plan: [] },
              },
            ],
          },
        ],
      },
    });
    expect(cleared?.todo_items).toEqual([]);
  });

  it("queued turn 不应进入普通历史投影", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-queue",
        sessionId: "session-queue",
        status: { type: "active", activeFlags: [] },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 5,
        turns: [
          {
            id: "turn-active",
            status: "inProgress",
            queue: { state: "running" },
            items: [],
          },
          {
            id: "turn-queued",
            status: "inProgress",
            queue: { state: "queued", position: 0 },
            items: [],
          },
        ],
      },
    });

    expect(detail?.messages).toEqual([]);
    expect(detail?.turns).toEqual([
      expect.objectContaining({ id: "turn-active", status: "running" }),
    ]);
  });

  it("保留 App Server parent-owned direct-input policy", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-child",
        sessionId: "session-child",
        parentThreadId: "thread-parent",
        canAcceptDirectInput: false,
        status: { type: "idle" },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS,
        turns: [],
      },
    });

    expect(detail?.thread_read).toMatchObject({
      thread_id: "thread-child",
      can_accept_direct_input: false,
    });
    expect(
      readCanonicalThreadListResponse({
        data: [
          {
            id: "thread-child",
            sessionId: "session-child",
            parentThreadId: "thread-parent",
            canAcceptDirectInput: false,
            createdAt: CREATED_AT_SECONDS,
            updatedAt: CREATED_AT_SECONDS,
            status: { type: "idle" },
          },
        ],
      }),
    ).toEqual([
      expect.objectContaining({
        parentThreadId: "thread-parent",
        canAcceptDirectInput: false,
      }),
    ]);
  });

  it("截断的 Codex MCP 结果仅保留空 content 时仍应恢复完整会话", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-codex-import-truncated-mcp",
        sessionId: "session-codex-import-truncated-mcp",
        status: { type: "idle" },
        createdAt: 1_700_000_000,
        updatedAt: 1_700_000_001,
        turns: [
          {
            id: "turn-codex-import-truncated-mcp",
            status: "completed",
            items: [
              {
                id: "mcp-codex-import-truncated",
                type: "mcpToolCall",
                server: "node_repl",
                tool: "exec",
                arguments: {},
                status: "failed",
                result: {
                  content: [],
                  _meta: { truncated: true },
                },
                error: { message: "tool output unavailable" },
              },
            ],
          },
        ],
      },
    });

    expect(detail).not.toBeNull();
    expect(detail?.items).toEqual([
      expect.objectContaining({
        id: "mcp-codex-import-truncated",
        status: "failed",
        type: "tool_call",
      }),
    ]);
  });

  it("按 Codex Unix 秒投影时间，并保留未加载状态", () => {
    const result = readCanonicalThreadListResponse({
      data: [
        {
          id: "thread-codex",
          sessionId: "session-codex",
          preview: "Codex thread",
          modelProvider: "openai",
          cwd: "/tmp/codex",
          createdAt: CREATED_AT_SECONDS,
          updatedAt: CREATED_AT_SECONDS + 2,
          status: { type: "notLoaded" },
        },
      ],
    });

    expect(result).toEqual([
      expect.objectContaining({
        createdAt: new Date(CREATED_AT_SECONDS * 1_000).toISOString(),
        updatedAt: new Date((CREATED_AT_SECONDS + 2) * 1_000).toISOString(),
        threadStatus: "unknown",
      }),
    ]);

    expect(
      readCanonicalThreadListResponse({
        data: [
          {
            threadId: "thread-alias",
            sessionId: "session-alias",
            createdAtMs: CREATED_AT_SECONDS * 1_000,
            updatedAtMs: CREATED_AT_SECONDS * 1_000,
            status: { type: "idle" },
          },
        ],
      }),
    ).toBeNull();
    expect(
      readCanonicalThreadListResponse({
        data: [
          {
            id: "thread-missing-status",
            sessionId: "session-missing-status",
            createdAt: CREATED_AT_SECONDS,
            updatedAt: CREATED_AT_SECONDS,
          },
        ],
      }),
    ).toBeNull();
  });

  it("按 Turn 与 Item 生命周期投影状态、失败原因和结构化工具结果", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-runtime",
        sessionId: "session-runtime",
        preview: "Runtime thread",
        modelProvider: "openai",
        cwd: "/tmp/runtime",
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 4,
        status: { type: "active", activeFlags: [] },
        turns: [
          {
            id: "turn-running",
            status: "inProgress",
            startedAt: CREATED_AT_SECONDS + 1,
            items: [
              {
                id: "item-user",
                type: "userMessage",
                content: [
                  { type: "text", text: "分析" },
                  { type: "image", url: "https://example.test/a.png" },
                  { type: "localImage", path: "/tmp/a.png" },
                ],
              },
              {
                id: "item-command",
                type: "commandExecution",
                command: "npm test",
                cwd: "/tmp/runtime",
                status: "inProgress",
                processId: "process-command",
                source: "agent",
                commandActions: [{ type: "read", path: "package.json" }],
                durationMs: 19,
              },
              {
                id: "item-mcp",
                type: "mcpToolCall",
                server: "files",
                tool: "read",
                arguments: { path: "README.md" },
                status: "failed",
                result: {
                  content: [{ type: "text", text: "partial" }],
                  structuredContent: { path: "README.md" },
                },
                error: { message: "permission denied" },
              },
            ],
          },
          {
            id: "turn-failed",
            status: "failed",
            startedAt: CREATED_AT_SECONDS + 2,
            completedAt: CREATED_AT_SECONDS + 3,
            error: { message: "provider failed" },
            items: [
              {
                id: "item-patch",
                type: "fileChange",
                changes: [{ path: "src/main.ts" }],
                status: "declined",
              },
              {
                id: "item-failed-message",
                type: "agentMessage",
                text: "partial answer",
              },
            ],
          },
          {
            id: "turn-completed",
            status: "completed",
            startedAt: CREATED_AT_SECONDS + 3,
            completedAt: CREATED_AT_SECONDS + 4,
            items: [
              {
                id: "item-completed-message",
                type: "agentMessage",
                text: "done",
              },
            ],
          },
        ],
      },
    });

    expect(detail).not.toBeNull();
    expect(detail?.turns).toEqual([
      expect.objectContaining({
        id: "turn-running",
        status: "running",
        prompt_text: "分析",
        started_at: new Date((CREATED_AT_SECONDS + 1) * 1_000).toISOString(),
        completed_at: undefined,
      }),
      expect.objectContaining({
        id: "turn-failed",
        status: "failed",
        error_message: "provider failed",
      }),
      expect.objectContaining({
        id: "turn-completed",
        status: "completed",
      }),
    ]);

    const command = detail?.items?.find((item) => item.id === "item-command");
    const mcp = detail?.items?.find((item) => item.id === "item-mcp");
    const patch = detail?.items?.find((item) => item.id === "item-patch");
    const failedMessage = detail?.items?.find(
      (item) => item.id === "item-failed-message",
    );
    const completedMessage = detail?.items?.find(
      (item) => item.id === "item-completed-message",
    );
    expect(command).toMatchObject({
      status: "in_progress",
      process_id: "process-command",
      source: "agent",
      command_actions: [{ type: "read", path: "package.json" }],
      duration_ms: 19,
    });
    expect(command).not.toHaveProperty("completed_at");
    expect(mcp).toMatchObject({
      status: "failed",
      success: false,
      error: "permission denied",
      structured_content: { path: "README.md" },
      metadata: expect.objectContaining({ server: "files" }),
    });
    expect(patch).toMatchObject({
      status: "completed",
      file_status: "declined",
      success: false,
    });
    expect(failedMessage).toMatchObject({ status: "completed" });
    expect(completedMessage).toMatchObject({ status: "completed" });

    expect(detail?.messages).toEqual([]);
    expect(
      detail?.items?.find((item) => item.id === "item-user"),
    ).toMatchObject({
      type: "user_message",
      content: "分析",
      content_parts: [
        { type: "text", text: "分析" },
        {
          type: "image",
          data: "",
          uri: "https://example.test/a.png",
        },
        {
          type: "image",
          data: "",
          display_name: "a.png",
          unavailable_reason: "host_reference_required",
        },
      ],
    });
  });

  it("按每个 Turn 的 items 原序生成 0-based sequence，并忽略上游 legacy sequence", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-item-sequence",
        sessionId: "session-item-sequence",
        status: { type: "idle" },
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 2,
        turns: [
          {
            id: "turn-item-sequence-first",
            status: "completed",
            startedAt: CREATED_AT_SECONDS,
            items: [
              {
                id: "item-sequence-first-user",
                type: "userMessage",
                content: [{ type: "text", text: "第一轮" }],
                sequence: 1_753_132_800_000,
              },
              {
                id: "item-sequence-first-agent",
                type: "agentMessage",
                text: "第一轮完成",
                sequence: 99,
              },
            ],
          },
          {
            id: "turn-item-sequence-second",
            status: "completed",
            startedAt: CREATED_AT_SECONDS + 1,
            items: [
              {
                id: "item-sequence-second-user",
                type: "userMessage",
                content: [{ type: "text", text: "第二轮" }],
                sequence: 1_753_132_900_000,
              },
              {
                id: "item-sequence-second-agent",
                type: "agentMessage",
                text: "第二轮完成",
                sequence: 100,
              },
            ],
          },
        ],
      },
    });

    expect(detail?.items?.map((item) => [item.id, item.sequence])).toEqual([
      ["item-sequence-first-user", 0],
      ["item-sequence-first-agent", 1],
      ["item-sequence-second-user", 0],
      ["item-sequence-second-agent", 1],
    ]);
  });

  it("遇到未知 Codex ThreadItem 时保留 thread/read 并投影安全诊断", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-unknown-item",
        sessionId: "session-unknown-item",
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS,
        status: { type: "idle" },
        turns: [
          {
            id: "turn-unknown-item",
            status: "completed",
            startedAt: CREATED_AT_SECONDS,
            completedAt: CREATED_AT_SECONDS,
            items: [
              {
                id: "item-unknown",
                type: "futureUnknownItem",
              },
            ],
          },
        ],
      },
    });

    expect(detail?.items).toEqual([
      expect.objectContaining({
        id: "item-unknown",
        type: "unknown_item",
        upstream_type: "futureUnknownItem",
        field_names: [],
      }),
    ]);
    expect(detail?.messages).toEqual([]);
  });

  it("从 v2 Thread.extra 投影文章工作台与用户可见 artifacts", () => {
    const detail = readCanonicalThreadDetail({
      thread: {
        id: "thread-article",
        sessionId: "session-article",
        createdAt: CREATED_AT_SECONDS,
        updatedAt: CREATED_AT_SECONDS + 1,
        status: { type: "idle" },
        extra: {
          articleWorkspace: {
            schemaVersion: "article-workspace.v1",
            appId: "content-factory-app",
            sessionId: "session-article",
            objects: [
              {
                ref: {
                  appId: "content-factory-app",
                  kind: "articleDraft",
                  id: "article-1",
                  sessionId: "session-article",
                },
              },
            ],
          },
          artifacts: [
            {
              artifactRef: "artifact-article-1",
              kind: "artifact_document",
            },
          ],
          workflowRuns: [{ workflowRunId: "internal-run" }],
        },
        turns: [],
      },
    });

    expect(detail?.thread_read).toMatchObject({
      articleWorkspace: {
        appId: "content-factory-app",
        sessionId: "session-article",
      },
      article_workspace: {
        appId: "content-factory-app",
        sessionId: "session-article",
      },
      artifacts: [
        {
          artifactRef: "artifact-article-1",
          kind: "artifact_document",
        },
      ],
    });
    expect(detail?.thread_read).not.toHaveProperty("workflowRuns");
    expect(detail?.thread_read).not.toHaveProperty("workflow_runs");
  });
});
