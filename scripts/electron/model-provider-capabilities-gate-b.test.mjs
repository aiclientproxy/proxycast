import fs from "node:fs";

import { describe, expect, it } from "vitest";

import {
  assertModelProviderCapabilitiesEvidence,
  EXPECTED_PROVIDER_CAPABILITIES,
  MODEL_PROVIDER_CAPABILITIES_METHOD,
  parseModelProviderCapabilitiesArgs,
  summarizeModelProviderCapabilitiesEvidence,
} from "./model-provider-capabilities-gate-b.mjs";

function traceRaw(transport = "electron-ipc", status = "success") {
  return JSON.stringify([
    {
      command: "app_server_handle_json_lines",
      transport,
      status,
      args_preview: {
        request: {
          lines: [
            JSON.stringify({
              jsonrpc: "2.0",
              id: "capabilities",
              method: MODEL_PROVIDER_CAPABILITIES_METHOD,
              params: {},
            }),
          ],
        },
      },
    },
  ]);
}

function visibleDom() {
  return {
    selectorVisible: true,
    panelVisible: true,
    badgeLabels: ["No tool namespaces", "Image generation", "Web search"],
    activeStates: EXPECTED_PROVIDER_CAPABILITIES,
    loadingVisible: false,
  };
}

describe("Model provider capabilities Electron Gate B", () => {
  it("accepts exact visible capability state over current Electron IPC", () => {
    const evidence = summarizeModelProviderCapabilitiesEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: visibleDom(),
    });
    expect(() =>
      assertModelProviderCapabilitiesEvidence(evidence),
    ).not.toThrow();
    expect(evidence.bridge.electronIpcHitCount).toBe(1);
    expect(evidence.bridge.mockFallbackHitCount).toBe(0);
  });

  it("fails closed for renderer mock transport or stale GUI values", () => {
    const mockEvidence = summarizeModelProviderCapabilitiesEvidence({
      traceRaw: traceRaw("renderer-mock"),
      errorRaw: "[]",
      dom: visibleDom(),
    });
    expect(() => assertModelProviderCapabilitiesEvidence(mockEvidence)).toThrow(
      /electron-ipc/,
    );

    const staleEvidence = summarizeModelProviderCapabilitiesEvidence({
      traceRaw: traceRaw(),
      errorRaw: "[]",
      dom: { ...visibleDom(), activeStates: [false, false, false] },
    });
    expect(() =>
      assertModelProviderCapabilitiesEvidence(staleEvidence),
    ).toThrow(/GUI 状态不正确/);
  });

  it("validates CLI bounds and keeps production mock paths absent", () => {
    expect(
      parseModelProviderCapabilitiesArgs(["--timeout-ms", "60000"], {
        evidenceDir: "/tmp/evidence",
        prefix: "provider-capabilities",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toMatchObject({ timeoutMs: 60_000 });
    expect(() =>
      parseModelProviderCapabilitiesArgs(["--prefix", "../unsafe"], {
        evidenceDir: "/tmp/evidence",
        prefix: "provider-capabilities",
        timeoutMs: 120_000,
        intervalMs: 250,
        keepTemp: false,
      }),
    ).toThrow(/invalid evidence prefix/);

    const content = fs.readFileSync(
      "scripts/electron/model-provider-capabilities-gate-b.mjs",
      "utf8",
    );
    expect(content).toContain("ensureElectronFixtureBuild");
    expect(content).toContain('backendMode: "unavailable"');
    expect(content).toContain('data-testid="model-selector"');
    expect(content).toContain(MODEL_PROVIDER_CAPABILITIES_METHOD);
    expect(content).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(content).not.toContain("mockPriorityCommands");
    expect(content).not.toContain("defaultMocks");
    expect(content).not.toContain("invokeMockOnly");
  });
});
