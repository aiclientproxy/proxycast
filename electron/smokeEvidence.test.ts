import { describe, expect, it } from "vitest";
import {
  buildElectronSmokeSummary,
  ELECTRON_SMOKE_LAYOUT_CLAIM_BOUNDARY,
  ELECTRON_SMOKE_LAYOUT_PROOF_LEVEL,
  ELECTRON_SMOKE_LAYOUT_VIEWPORTS,
  isElectronSmokeStartupUrl,
  normalizeElectronSmokeRunId,
  sanitizeElectronSmokeLocation,
  type ElectronSmokeLayoutEvidence,
  type ElectronSmokeSummaryInput,
} from "./smokeEvidence";

function passingLayout(): ElectronSmokeLayoutEvidence {
  const rect = {
    x: 0,
    y: 0,
    width: 100,
    height: 100,
    right: 100,
    bottom: 100,
  };
  const nodes = Object.fromEntries(
    [
      ["workspaceShell", "workspace-shell-scene", true],
      ["workspaceMainArea", "workspace-main-area", true],
      ["inputbarCore", "inputbar-core-container", true],
      ["composerInput", "agent-chat-message", true],
      ["messageListColumn", "message-list-column", false],
      ["emptyStateFirstScreen", "empty-state-first-screen", true],
      ["threadHeader", "thread-workspace-header", false],
      ["threadHeaderTitle", "thread-workspace-header-title", false],
      ["threadHeaderActions", "thread-workspace-header-actions", false],
    ].map(([key, testId, required]) => [
      key,
      {
        testId,
        present: true,
        visible: true,
        required,
        rect,
        withinViewport: true,
      },
    ]),
  ) as ElectronSmokeLayoutEvidence["viewports"][number]["nodes"];
  const viewports = ELECTRON_SMOKE_LAYOUT_VIEWPORTS.map((requested) => ({
    requested,
    window: requested,
    viewport: requested,
    document: {
      scrollWidth: requested.width,
      scrollHeight: requested.height,
    },
    nodes,
    assertions: {
      windowSizeMatchesRequest: true,
      requiredNodesVisible: true,
      requiredNodesWithinViewport: true,
      contentSurfacePresent: true,
      noHorizontalOverflow: true,
      headerTitleVisible: true,
      headerActionsDoNotOverlap: true,
    },
  }));
  return {
    proofLevel: ELECTRON_SMOKE_LAYOUT_PROOF_LEVEL,
    claimBoundary: ELECTRON_SMOKE_LAYOUT_CLAIM_BOUNDARY,
    viewports,
    screenshots: ELECTRON_SMOKE_LAYOUT_VIEWPORTS.map(
      ({ width, height }) => `layout-${width}x${height}.png`,
    ),
    assertions: {
      expectedViewportCount: ELECTRON_SMOKE_LAYOUT_VIEWPORTS.length,
      capturedViewportCount: ELECTRON_SMOKE_LAYOUT_VIEWPORTS.length,
      allViewportsPass: true,
      composerHeightStable: true,
      composerHeightRange: 0,
    },
  };
}

function passingInput(): ElectronSmokeSummaryInput {
  return {
    runId: "candidate-20260716",
    startedAt: "2026-07-16T00:00:00.000Z",
    completedAt: "2026-07-16T00:00:02.000Z",
    appVersion: "0.1.0",
    backendMode: "unavailable",
    hostAppServerInitialized: true,
    hostAppServerProtocol: "v0",
    routes: [
      {
        stage: "startup",
        ready: true,
        location: "file:///main-window-startup.html",
      },
      {
        stage: "workbench",
        ready: true,
        location: "http://127.0.0.1:1420/?nativeStartup",
      },
      {
        stage: "workbench-reload",
        ready: true,
        location: "http://127.0.0.1:1420/?nativeStartup",
      },
      {
        stage: "settings-memory",
        ready: true,
        location: "http://127.0.0.1:1420/?nativeStartup",
      },
    ],
    renderer: {
      electron: true,
      preloadInvoke: true,
      appServerCommandSupported: true,
      appServerIpcHitCount: 3,
      appServerMethods: ["memoryStore/status", "memoryStore/review/list"],
      invokeErrorCount: 0,
      traceErrorCount: 0,
      legacyCommandHitCount: 0,
      legacyCommands: [],
      mockFallbackHitCount: 0,
      pageErrorCount: 0,
    },
    layout: passingLayout(),
    diagnostics: {
      consoleErrorCount: 0,
      rendererCrashCount: 0,
      rendererUnresponsiveCount: 0,
      preloadErrorCount: 0,
      rendererLoadErrorCount: 0,
    },
    artifacts: {
      summary: "summary.json",
      trace: "trace-summary.json",
      screenshot: "settings-memory.png",
      screenshotCaptured: true,
    },
  };
}

describe("electron smoke evidence", () => {
  it("builds a passing Gate B-F shell summary without request payloads", () => {
    const summary = buildElectronSmokeSummary(passingInput());

    expect(summary.result).toBe("pass");
    expect(summary.proofLevel).toBe("Gate B-F");
    expect(summary.bridge.transport).toBe("electron-ipc");
    expect(summary.bridge.methods).toEqual([
      "memoryStore/review/list",
      "memoryStore/status",
    ]);
    expect(summary.assertions.failed).toEqual([]);
    expect(summary.assertions.details.traceCaptured).toBe(true);
    expect(summary.assertions.details.layoutViewportsCaptured).toBe(true);
    expect(summary.assertions.details.layoutGeometryStable).toBe(true);
    expect(summary.assertions.details.layoutScreenshotsCaptured).toBe(true);
    expect(summary.surfaceProof).toEqual({
      surfaceId: "SHELL-01",
      proof: "gate-b-f",
      complete: true,
    });
    expect(JSON.stringify(summary)).not.toContain("params");
    expect(JSON.stringify(summary)).not.toContain('"request":');
  });

  it("fails closed when bridge, route, or error assertions are missing", () => {
    const input = passingInput();
    input.routes = input.routes.filter((route) => route.stage !== "startup");
    input.renderer.appServerIpcHitCount = 0;
    input.renderer.appServerMethods = [];
    input.diagnostics.consoleErrorCount = 1;
    input.artifacts.trace = null;
    input.layout.assertions.allViewportsPass = false;

    const summary = buildElectronSmokeSummary(input);

    expect(summary.result).toBe("fail");
    expect(summary.failedStage).toBe("contract-assertions");
    expect(summary.failureClass).toBe("product");
    expect(summary.nextAction).toMatch(/rerun/);
    expect(summary.assertions.failed).toEqual(
      expect.arrayContaining([
        "startupVisible",
        "electronIpcAppServerBridgeUsed",
        "currentAppServerMethodObserved",
        "noConsoleErrors",
        "traceCaptured",
        "layoutGeometryStable",
      ]),
    );
  });

  it("fails closed when a viewport geometry assertion fails", () => {
    const input = passingInput();
    input.layout.viewports[2].assertions.noHorizontalOverflow = false;
    input.layout.assertions.allViewportsPass = false;

    const summary = buildElectronSmokeSummary(input);

    expect(summary.result).toBe("fail");
    expect(summary.assertions.failed).toContain("layoutGeometryStable");
  });

  it("normalizes run ids and removes local path and query values", () => {
    expect(normalizeElectronSmokeRunId(" gate-a_1 ", "fallback")).toBe(
      "gate-a_1",
    );
    expect(() => normalizeElectronSmokeRunId("bad/id", "fallback")).toThrow(
      /LIME_GATE_RUN_ID/,
    );
    expect(
      sanitizeElectronSmokeLocation(
        "file:///Users/example/private/index.html?nativeStartup=1&token=secret",
      ),
    ).toBe("file:///index.html?nativeStartup&token");
    expect(
      sanitizeElectronSmokeLocation(
        "http://127.0.0.1:1420/workspace?nativeStartup=1",
      ),
    ).toBe("http://127.0.0.1:1420/workspace?nativeStartup");
  });

  it("recognizes both file and data startup documents", () => {
    expect(
      isElectronSmokeStartupUrl(
        "file:///tmp/profile/startup/main-window-startup.html",
      ),
    ).toBe(true);
    expect(isElectronSmokeStartupUrl("data:text/html,<main></main>")).toBe(
      true,
    );
    expect(
      isElectronSmokeStartupUrl("http://127.0.0.1:1420/?nativeStartup=1"),
    ).toBe(false);
  });
});
