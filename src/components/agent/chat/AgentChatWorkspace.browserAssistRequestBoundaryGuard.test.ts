import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace Browser intent boundary", () => {
  it("Browser artifact 只能请求 Right Surface，不能恢复 session/CDP owner", () => {
    const ownerPath = join(
      process.cwd(),
      "src/components/agent/chat/workspace/useWorkspaceBrowserAssistRequestRuntime.ts",
    );
    const ownerSource = readFileSync(ownerPath, "utf8");

    expect(ownerSource).toContain("requestWorkspaceRightSurface({");
    expect(ownerSource).toContain('surfaceKind: "browser"');
    expect(ownerSource).not.toMatch(
      /BrowserAssistSessionState|BrowserSessionRef|profileKey|preferredBackend|autoLaunch|browserExecuteAction|onNavigate/,
    );

    for (const retiredPath of [
      "src/components/agent/chat/workspace/useWorkspaceBrowserAssistCanvasRuntime.ts",
      "src/components/agent/chat/workspace/useWorkspaceBrowserAssistRuntimeCore.ts",
      "src/components/agent/chat/workspace/workspaceBrowserSessionRef.ts",
      "src/components/agent/chat/workspace/workspaceBrowserRuntimeNavigation.ts",
    ]) {
      expect(existsSync(join(process.cwd(), retiredPath))).toBe(false);
    }
  });
});
