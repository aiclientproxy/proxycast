import { describe, expect, it, vi } from "vitest";
import { buildRightSurfaceState } from "./right-surface";
import {
  buildWorkspaceConversationRightSurfaceChrome,
  buildWorkspaceConversationRightSurfaceSceneProps,
} from "./workspaceConversationRightSurfaceChrome";

function createRightSurfaceRuntime(
  activeSurface: Parameters<typeof buildRightSurfaceState>[0],
) {
  return {
    rightSurfaceLaunchers: [
      {
        kind: "browser" as const,
        label: "浏览器",
        active: activeSurface === "browser",
        pendingCount: 0,
        disabled: false,
        collapseTarget: "topToolbar" as const,
      },
    ],
    rightSurfaceState: buildRightSurfaceState(activeSurface, "user"),
    handleToggleRightSurfaceBrowser: vi.fn(),
    handleToggleRightSurfaceFiles: vi.fn(),
    handleToggleRightSurfaceTrace: vi.fn(),
    handleToggleRightSurfaceShell: vi.fn(),
    handleToggleRightSurfaceHarness: vi.fn(),
    handleToggleExpertInfoPanel: vi.fn(),
  };
}

describe("workspaceConversationRightSurfaceChrome", () => {
  it("应从 right surface runtime 派生 chrome active 状态和动作", () => {
    const runtime = createRightSurfaceRuntime("browser");

    const chrome = buildWorkspaceConversationRightSurfaceChrome({
      content: "right-surface",
      rightSurfaceRuntime: runtime,
      showHarnessToggle: true,
      hasExpertInfoPanel: true,
      expertInfoPanelVisible: false,
      harnessPendingCount: 2,
      harnessAttentionLevel: "warning",
      harnessToggleLabel: "Harness",
    });

    expect(chrome.content).toBe("right-surface");
    expect(chrome.launchers).toBe(runtime.rightSurfaceLaunchers);
    expect(chrome.browserOpen).toBe(true);
    expect(chrome.filesOpen).toBe(false);
    expect(chrome.onToggleBrowser).toBe(
      runtime.handleToggleRightSurfaceBrowser,
    );
    expect(chrome.showHarnessToggle).toBe(true);
    expect(chrome.harnessPanelVisible).toBe(false);
    expect(chrome.showExpertInfoToggle).toBe(true);
    expect(chrome.harnessPendingCount).toBe(2);
    expect(chrome.harnessAttentionLevel).toBe("warning");
  });

  it("隐藏 harness chrome 时只关闭 harness 展示，不影响右侧 surface 内容", () => {
    const runtime = createRightSurfaceRuntime("harness");

    const chrome = buildWorkspaceConversationRightSurfaceChrome({
      content: "right-surface",
      rightSurfaceRuntime: runtime,
      showHarnessToggle: true,
      hasExpertInfoPanel: false,
      harnessPendingCount: 3,
      harnessAttentionLevel: "active",
      harnessToggleLabel: "Harness",
      suppressHarnessChrome: true,
    });

    expect(chrome.content).toBe("right-surface");
    expect(chrome.showHarnessToggle).toBe(false);
    expect(chrome.harnessPanelVisible).toBe(false);
    expect(chrome.harnessPendingCount).toBe(0);
    expect(chrome.harnessAttentionLevel).toBe("idle");
    expect(chrome.harnessToggleLabel).toBeUndefined();
  });

  it("scene props 在隐藏 utility actions 时保留内容但关闭右侧按钮动作", () => {
    const runtime = createRightSurfaceRuntime("files");
    const chrome = buildWorkspaceConversationRightSurfaceChrome({
      content: "right-surface",
      rightSurfaceRuntime: runtime,
      showHarnessToggle: true,
      hasExpertInfoPanel: true,
      expertInfoPanelVisible: true,
      harnessPendingCount: 1,
      harnessAttentionLevel: "active",
      harnessToggleLabel: "Harness",
    });

    const sceneProps = buildWorkspaceConversationRightSurfaceSceneProps({
      rightSurfaceChrome: chrome,
      utilityActionsVisible: false,
    });

    expect(sceneProps.rightSurfaceContent).toBe("right-surface");
    expect(sceneProps.rightSurfaceFilesOpen).toBe(false);
    expect(sceneProps.onToggleRightSurfaceFiles).toBeUndefined();
    expect(sceneProps.showHarnessToggle).toBe(false);
    expect(sceneProps.harnessPanelVisible).toBe(false);
    expect(sceneProps.harnessPendingCount).toBe(0);
    expect(sceneProps.harnessAttentionLevel).toBe("idle");
    expect(sceneProps.harnessToggleLabel).toBeUndefined();
  });
});
