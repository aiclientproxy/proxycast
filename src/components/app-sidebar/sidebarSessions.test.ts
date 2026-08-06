import { describe, expect, it } from "vitest";
import type { AgentSessionInfo } from "@/lib/api/agentRuntime/sessionTypes";
import {
  buildImportedSidebarSession,
  filterSidebarSessions,
} from "./sidebarSessions";

function session(
  id: string,
  overrides: Partial<AgentSessionInfo> = {},
): AgentSessionInfo {
  return {
    id,
    name: id,
    created_at: 1,
    updated_at: 1,
    archived_at: null,
    ...overrides,
  };
}

describe("sidebarSessions", () => {
  it("保持 App Server 顺序，并容忍缺少 id 或时间字段的记录", () => {
    const staleShape = {
      name: "旧形状会话",
      archived_at: null,
    } as unknown as AgentSessionInfo;

    expect(() =>
      filterSidebarSessions([
        session("older", { updated_at: 10, created_at: 10 }),
        staleShape,
        session("newer", { updated_at: 20, created_at: 20 }),
      ]),
    ).not.toThrow();

    expect(
      filterSidebarSessions([
        session("older", { updated_at: 10, created_at: 10 }),
        staleShape,
        session("newer", { updated_at: 20, created_at: 20 }),
      ]).map((item) => item.id || item.name),
    ).toEqual(["older", "旧形状会话", "newer"]);
  });

  it("把导入结果投影为侧栏会话项", () => {
    const session = buildImportedSidebarSession({
      session: {
        sessionId: "session-imported",
        threadId: "thread-imported",
        appId: "content-studio",
        workspaceId: "project-1",
        status: "completed",
        createdAt: "2026-06-15T00:00:00.000Z",
        updatedAt: "2026-06-16T00:00:01.000Z",
      },
      thread: {
        sourceClient: "codex",
        sourceThreadId: "thread-1",
        title: "本地历史修复记录",
        createdAt: "2026-06-14T00:00:00.000Z",
        updatedAt: "2026-06-15T00:00:00.000Z",
        cwd: "/repo/project-1",
        source: "cli",
        modelProvider: "openai",
        archived: false,
        sourcePath: "/Users/example/.codex/sessions/thread-1.jsonl",
        importStatus: "imported",
      },
      summary: {
        lineCount: 8,
        messageCount: 2,
        rolloutEventItems: 2,
        unsupportedCount: 0,
        dryRun: {
          willCreateSession: true,
          willAppendToExistingSession: false,
          willImportMessages: 2,
          willImportTurns: 1,
          willImportTimelineItems: 4,
          willImportAttachments: 0,
          unsupportedItems: 0,
        },
        fidelity: {
          messages: 2,
          reasoning: 0,
          tools: 0,
          commands: 0,
          patches: 0,
          approvals: 0,
          mcp: 0,
          webSearch: 0,
          attachments: 0,
          unsupported: 0,
          provenanceOnly: 0,
          budgetDropped: 0,
        },
        truncated: false,
        warnings: [],
      },
      importedMessages: 2,
      importedTurns: 1,
      canContinue: true,
      warnings: [],
    });

    expect(session).toEqual({
      id: "session-imported",
      name: "本地历史修复记录",
      created_at: 1781481600,
      updated_at: 1781568001,
      archived_at: null,
      model: "openai",
      messages_count: 2,
      workspace_id: "project-1",
      working_dir: "/repo/project-1",
    });
  });
});
