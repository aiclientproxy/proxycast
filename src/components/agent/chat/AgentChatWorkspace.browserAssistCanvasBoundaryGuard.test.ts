import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace Browser Canvas negative guard", () => {
  it("通用 Artifact/Canvas 不得重新承载 Browser runtime", () => {
    const artifactOwnerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactCanvasRuntime.ts",
      ),
      "utf8",
    );
    const layoutSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceCanvasLayoutRuntime.ts",
      ),
      "utf8",
    );

    expect(artifactOwnerSource).toContain(
      "useWorkspaceBrowserAssistRequestRuntime({",
    );
    expect(`${artifactOwnerSource}\n${layoutSource}`).not.toMatch(
      /useWorkspaceBrowserAssistCanvasRuntime|browserAssistCanvasControl|BrowserSessionRef|browserExecuteAction/,
    );
    expect(
      existsSync(
        join(
          process.cwd(),
          "src/components/agent/chat/components/canvas-workbench/browser/CanvasWorkbenchBrowserPanel.tsx",
        ),
      ),
    ).toBe(false);
  });
});
