import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const SESSION_STATE_SOURCE =
  "src/components/agent/chat/hooks/agentSessionState.ts";
const SESSION_HOOK_SOURCE =
  "src/components/agent/chat/hooks/useAgentSession.ts";
const CHAT_HOOK_SOURCE = "src/components/agent/chat/hooks/useAgentChat.ts";
const WORKSPACE_SOURCE =
  "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx";
const WORKSPACE_SHARED_SOURCE =
  "src/components/agent/chat/agentChatWorkspaceShared.ts";

function readSource(relativePath: string): string {
  return readFileSync(join(process.cwd(), relativePath), "utf8");
}

describe("React canonical roster plumbing boundary", () => {
  it("session state、hooks 与 Workspace 不得重新保留 legacy roster", () => {
    const combined = [
      SESSION_STATE_SOURCE,
      SESSION_HOOK_SOURCE,
      CHAT_HOOK_SOURCE,
      WORKSPACE_SOURCE,
      WORKSPACE_SHARED_SOURCE,
    ]
      .map(readSource)
      .join("\n");

    expect(combined).not.toContain("childSubagentSessions");
    expect(combined).not.toContain("AgentSubagentSessionInfo");
    expect(combined).not.toContain("detail.child_subagent_sessions");
    expect(combined).not.toContain("subagentParentContext");
    expect(combined).not.toContain("AgentSubagentParentContext");
    expect(combined).not.toContain("detail.subagent_parent_context");
    expect(combined).not.toContain("deriveCurrentSessionRuntimeStatus");
    expect(combined).not.toContain("deriveLatestTurnRuntimeStatus");
  });

  it("Workspace 继续消费 canonical child roster 与 parent identity", () => {
    const workspaceSource = readSource(WORKSPACE_SOURCE);
    const teamRuntimeSource = readSource(
      "src/components/agent/chat/workspace/useWorkspaceTeamRuntime.ts",
    );

    expect(workspaceSource).toContain("canonicalChildren,");
    expect(teamRuntimeSource).toContain(
      "hasParentThread: canonical.hasParentThread,",
    );
    expect(teamRuntimeSource).not.toContain("session.hasParentThread");
  });
});
