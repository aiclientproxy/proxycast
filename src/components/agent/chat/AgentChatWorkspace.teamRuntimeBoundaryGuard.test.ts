import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace team runtime boundary", () => {
  it("Renderer 不得恢复本地 Team formation 或 dispatch preview", () => {
    const workspaceSource = [
      "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceEntryRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceCommandRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
    ]
      .map((ownerPath) => readFileSync(join(process.cwd(), ownerPath), "utf8"))
      .join("\n");
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceTeamRuntime.ts",
      ),
      "utf8",
    );
    const sendSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceSendActions.ts",
      ),
      "utf8",
    );
    const helperSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/workspaceSendHelpers.ts",
      ),
      "utf8",
    );
    const contractSource = readFileSync(
      join(process.cwd(), "packages/agent-ui-contracts/src/events.ts"),
      "utf8",
    );
    const projectionSummarySource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/projection/agentUiProjectionSummary.ts",
      ),
      "utf8",
    );
    const navigationSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceSubagentNavigationRuntime.ts",
      ),
      "utf8",
    );

    expect(workspaceSource).toContain("useWorkspaceTeamRuntime({");
    expect(ownerSource.split("\n").length).toBeLessThan(80);
    expect(ownerSource).toContain("useCanonicalChildThreads({");
    expect(ownerSource).toContain("canonicalChildren: canonical.children");
    expect(ownerSource).not.toContain("childSubagentSessions");
    expect(ownerSource).not.toContain("subagentParentContext");
    expect(ownerSource).toContain("handleStopSending: stopSending");
    expect(workspaceSource).toContain(
      "useWorkspaceSubagentNavigationRuntime({",
    );
    expect(workspaceSource).toContain("referencedChildThreadIds:");
    expect(workspaceSource).toContain("canonicalChildren,");
    expect(navigationSource).toContain("const canonicalSessionId");
    expect(navigationSource).toContain(
      "await readSessionId(normalizedTargetId)",
    );
    expect(navigationSource).not.toContain("childSubagentSessions");
    expect(navigationSource).not.toContain("isKnownSession");
    expect(navigationSource).toContain("await switchTopic(sessionId");
    expect(navigationSource).toContain("allowDetachedSession: true");

    const productionSource = [
      workspaceSource,
      ownerSource,
      sendSource,
      helperSource,
      contractSource,
      projectionSummarySource,
    ].join("\n");
    for (const retiredFormationSurface of [
      "useRuntimeTeamFormation",
      "prepareRuntimeTeamBeforeSend",
      "runtimeTeamDispatchPreview",
      "teamFormationAgentUiProjection",
      "team_formation_projection",
      "runtime-team-dispatch:",
      "useTeamWorkspaceRuntime",
      "useWorkspaceTeamSessionRuntime",
      "useWorkspaceTeamSessionControlRuntime",
      "listenSubagentStatus",
      "listenSubagentStream",
      "liveRuntimeBySessionId",
    ]) {
      expect(productionSource).not.toContain(retiredFormationSurface);
    }

    for (const deletedPath of [
      "src/components/agent/chat/hooks/useRuntimeTeamFormation.ts",
      "src/components/agent/chat/projection/teamFormationAgentUiProjection.ts",
      "src/components/agent/chat/workspace/runtimeTeamCollaborationCopy.ts",
      "src/components/agent/chat/hooks/useTeamWorkspaceRuntime.ts",
      "src/components/agent/chat/teamWorkspaceRuntime.ts",
      "src/components/agent/chat/team-workspace-runtime/liveRuntimeProjector.ts",
      "src/components/agent/chat/team-workspace-runtime/runtimeEventSubscriptions.ts",
      "src/components/agent/chat/workspace/useWorkspaceTeamSessionRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceTeamSessionControlRuntime.ts",
    ]) {
      expect(existsSync(join(process.cwd(), deletedPath))).toBe(false);
    }
    const retiredRuntimeDirectory = join(
      process.cwd(),
      "src/components/agent/chat/team-workspace-runtime",
    );
    expect(
      existsSync(retiredRuntimeDirectory)
        ? readdirSync(retiredRuntimeDirectory)
        : [],
    ).toEqual([]);
  });
});
