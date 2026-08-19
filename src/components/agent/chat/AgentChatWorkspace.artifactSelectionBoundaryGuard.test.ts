import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace artifact canvas runtime boundary", () => {
  it("artifact selection/store/view-mode 组合必须由 artifact canvas runtime 提供", () => {
    const workspaceSource = [
      "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceEntryRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceCommandRuntime.ts",
      "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
    ]
      .map((ownerPath) => readFileSync(join(process.cwd(), ownerPath), "utf8"))
      .join("\n");
    const canvasOwnerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactCanvasRuntime.ts",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactSelectionRuntime.ts",
      ),
      "utf8",
    );

    expect(workspaceSource).toContain("useWorkspaceArtifactCanvasRuntime({");
    expect(canvasOwnerSource.split("\n").length).toBeLessThan(180);
    for (const retiredWorkspaceArtifactCanvasRuntime of [
      "useWorkspaceArtifactSelectionRuntime({",
      "useWorkspaceArtifactStoreRuntime({",
      "useWorkspaceArtifactViewModeControl({",
    ]) {
      expect(workspaceSource).not.toContain(
        retiredWorkspaceArtifactCanvasRuntime,
      );
      expect(canvasOwnerSource).toContain(
        retiredWorkspaceArtifactCanvasRuntime,
      );
    }

    expect(ownerSource.split("\n").length).toBeLessThan(180);
    for (const retiredArtifactSelectionOwner of [
      "artifactsAtom",
      "selectedArtifactAtom",
      "selectedArtifactIdAtom",
      "hasNamedGeneralCanvasFilePreview",
      "resolveDefaultSelectedArtifact(",
      "const liveArtifact = useMemo(",
    ]) {
      expect(workspaceSource).not.toContain(retiredArtifactSelectionOwner);
      expect(ownerSource).toContain(retiredArtifactSelectionOwner);
    }
    expect(ownerSource).not.toContain("GENERAL_BROWSER_ASSIST_ARTIFACT_ID");
  });
});
