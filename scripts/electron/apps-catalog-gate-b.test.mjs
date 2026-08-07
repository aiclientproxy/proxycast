import fs from "node:fs";
import { describe, expect, it } from "vitest";

describe("Apps catalog Gate B guard", () => {
  it("uses the current Electron/App Server chain and visible App Center readiness", () => {
    const content = fs.readFileSync(
      "scripts/electron/apps-catalog-gate-b.mjs",
      "utf8",
    );

    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain("launchElectronFixture");
    expect(content).toContain('backendMode: "unavailable"');
    expect(content).toContain('"plugin/list"');
    expect(content).toContain('"plugin/install"');
    expect(content).toContain('"plugin/enabled/set"');
    expect(content).toContain('"app/list"');
    expect(content).toContain('"app/read"');
    expect(content).toContain('"app/installed"');
    expect(content).toContain('"app/list/updated"');
    expect(content).toContain('data-testid="app-sidebar-nav-plugins"');
    expect(content).toContain("plugin-v2-app-readiness-");
    expect(content).toContain("notificationFreshReadObserved");
    expect(content).toContain("callable: false");
    expect(content).toContain("mockFallbackHitCount");
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("window.dispatchEvent");
  });
});
