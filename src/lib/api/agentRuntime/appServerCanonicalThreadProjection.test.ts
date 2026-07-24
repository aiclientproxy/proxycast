import { describe, expect, it } from "vitest";

import {
  readCanonicalThreadDetail,
  readCanonicalThreadListResponse,
} from "./appServerCanonicalThreadProjection";

const CREATED_AT_SECONDS = 1_780_704_000;

describe("appServerCanonicalThreadProjection", () => {
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
      status: "failed",
      file_status: "declined",
      success: false,
    });
    expect(failedMessage).toMatchObject({ status: "completed" });
    expect(completedMessage).toMatchObject({ status: "completed" });

    expect(detail?.messages[0]).toMatchObject({
      role: "user",
      timestamp: CREATED_AT_SECONDS + 1,
      content: [
        { type: "text", text: "分析" },
        {
          type: "image",
          data: "",
          uri: "https://example.test/a.png",
        },
        {
          type: "image",
          data: "",
          uri: "/tmp/a.png",
          source_path: "/tmp/a.png",
        },
      ],
    });
  });

  it("遇到未知 Codex ThreadItem 时整个 thread/read fail closed", () => {
    expect(
      readCanonicalThreadDetail({
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
      }),
    ).toBeNull();
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
