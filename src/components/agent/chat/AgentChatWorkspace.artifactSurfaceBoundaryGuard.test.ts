import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace artifact surface runtime boundary", () => {
  it("artifact 交互必须由 artifact surface runtime 提供", () => {
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
        "src/components/agent/chat/workspace/useWorkspaceArtifactSurfaceRuntime.ts",
      ),
      "utf8",
    );
    const interactionOwnerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceArtifactInteractionRuntime.ts",
      ),
      "utf8",
    );

    expect(workspaceSource).toContain(
      "useAgentChatWorkspaceArtifactInteractionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceArtifactSurfaceRuntime({",
    );
    expect(interactionOwnerSource).toContain(
      "useWorkspaceArtifactSurfaceRuntime({",
    );
    expect(ownerSource.split("\n").length).toBeLessThan(180);
    const timelineJumpOwner = "const handleJumpToTimelineItem = useCallback(";
    expect(workspaceSource).not.toContain(timelineJumpOwner);
    expect(interactionOwnerSource).not.toContain(timelineJumpOwner);
    expect(ownerSource).toContain(timelineJumpOwner);
    expect(workspaceSource).not.toContain(
      "useWorkspaceServiceSkillExecutionCardRuntime(",
    );
    expect(interactionOwnerSource).not.toContain(
      "useWorkspaceServiceSkillExecutionCardRuntime(",
    );
    expect(ownerSource).not.toContain(
      "useWorkspaceServiceSkillExecutionCardRuntime(",
    );
  });
});
