import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();

function read(relativePath: string): string {
  return readFileSync(join(REPO_ROOT, relativePath), "utf8");
}

describe("Browser current owner boundary", () => {
  it("current Browser surface 不得恢复旧 MCP、Canvas 或 BrowserSessionRef owner", () => {
    const currentPaths = [
      "electron/browserTabHost.ts",
      "electron/appServerDynamicToolHost.ts",
      "src/lib/api/browserTab.ts",
      "src/components/agent/chat/utils/generalAgentPrompt.ts",
      "src/components/agent/chat/workspace/right-surface/browser/BrowserWorkspace.tsx",
      "src/components/agent/chat/workspace/right-surface/browser/RightSurfaceBrowserPanel.tsx",
    ];
    const forbidden = [
      "mcp__lime-browser__",
      "mcp__lime-browser",
      "browserSession/",
      "BrowserSessionRef",
      "CanvasWorkbenchBrowserPanel",
    ];

    const offenders = currentPaths.flatMap((path) => {
      const source = read(path);
      return forbidden
        .filter((token) => source.includes(token))
        .map((token) => ({ path, token }));
    });

    expect(offenders).toEqual([]);
  });

  it("旧 Canvas Browser owner 必须保持物理删除", () => {
    expect(
      existsSync(
        join(
          REPO_ROOT,
          "src/components/agent/chat/components/canvas-workbench/browser/CanvasWorkbenchBrowserPanel.tsx",
        ),
      ),
    ).toBe(false);
  });
});
