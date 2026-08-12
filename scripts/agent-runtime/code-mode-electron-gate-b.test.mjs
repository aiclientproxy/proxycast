import fs from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

import {
  buildAssertions,
  parseArgs,
  readCodeModeProcessEvidence,
  summarizeProviderEvidence,
} from "./code-mode-electron-gate-b.mjs";

describe("CodeMode Electron Gate B", () => {
  it("parses isolated evidence and polling options", () => {
    expect(
      parseArgs([
        "--output",
        ".lime/qc/code-mode.json",
        "--timeout-ms",
        "60000",
        "--interval-ms",
        "200",
      ]),
    ).toMatchObject({
      output: path.resolve(".lime/qc/code-mode.json"),
      timeoutMs: 60_000,
      intervalMs: 200,
    });
    expect(parseArgs(["--help"]).help).toBe(true);
    expect(() => parseArgs(["--timeout-ms", "1000"])).toThrow(
      "--timeout-ms must be >= 30000",
    );
    expect(() => parseArgs(["--interval-ms", "10"])).toThrow(
      "--interval-ms must be >= 100",
    );
    expect(() => parseArgs(["--unknown"])).toThrow(
      "Unknown argument: --unknown",
    );
  });

  it("summarizes the official-host custom exec round trip", () => {
    const evidence = summarizeProviderEvidence({
      modelRequests: [{ host: "api.openai.com" }],
      requests: [
        {
          path: "/v1/responses",
          host: "api.openai.com",
          body: {
            tools: [
              {
                type: "custom",
                name: "exec",
                format: { type: "grammar", syntax: "lark" },
              },
            ],
          },
        },
        {
          path: "/v1/responses",
          host: "api.openai.com",
          body: {
            input: [
              {
                type: "custom_tool_call_output",
                call_id: "call-code-mode-gate-b",
                output: "CODE_MODE_GATE_B_OK",
              },
            ],
          },
        },
      ],
    });

    expect(evidence).toMatchObject({
      discoveryRequestCount: 1,
      discoveryUsedOfficialHost: true,
      responsesRequestCount: 2,
      responsesUsedOfficialHost: true,
      firstRequestAdvertisedExec: true,
      execToolType: "custom",
      execFormatType: "grammar",
      execFormatSyntax: "lark",
      secondRequestHasCustomToolOutput: true,
      customToolOutputContainsMarker: true,
      requestErrors: [],
    });
  });

  it("requires every product-chain and visible-terminal assertion", () => {
    const assertions = buildAssertions({
      diagnostics: {
        calls: [
          {
            method: "turn/start",
            transport: "electron-ipc",
            status: "success",
          },
        ],
        invokeErrorCount: 0,
        mockFallbackHitCount: 0,
      },
      errors: { console: [], page: [] },
      model: {
        toolMode: "code_mode",
        runtimeFeatures: ["custom_tools", "responses_api"],
      },
      outerExec: { name: "exec", status: "completed" },
      processEvidence: {
        electronPid: 100,
        appServerPid: 101,
        codeModeHostPid: 102,
        codeModeHostParentPid: 101,
      },
      provider: {
        discoveryUsedOfficialHost: true,
        responsesUsedOfficialHost: true,
        firstRequestAdvertisedExec: true,
        execToolType: "custom",
        execFormatType: "grammar",
        execFormatSyntax: "lark",
        secondRequestHasCustomToolOutput: true,
        customToolOutputContainsMarker: true,
        requestErrors: [],
      },
      rendererSnapshot: {
        electron: true,
        hasInvokeBridge: true,
        supportsAppServer: true,
      },
      thread: { marker: "CODE_MODE_GATE_B_VISIBLE" },
      turn: {
        status: "completed",
        items: [{ type: "Tool", name: "exec", status: "completed" }],
      },
      visible: {
        finalAssistantTextVisible: true,
        toolRows: [{ name: "exec", status: "completed", visible: true }],
      },
    });

    expect(Object.values(assertions).every(Boolean)).toBe(true);
    expect(
      buildAssertions({
        diagnostics: {
          calls: [],
          invokeErrorCount: 0,
          mockFallbackHitCount: 0,
        },
        errors: { console: [], page: [] },
        model: { toolMode: "code_mode", runtimeFeatures: ["custom_tools"] },
        outerExec: { name: "exec", status: "completed" },
        processEvidence: {
          electronPid: 100,
          appServerPid: 101,
          codeModeHostPid: 102,
          codeModeHostParentPid: 101,
        },
        provider: {
          discoveryUsedOfficialHost: true,
          responsesUsedOfficialHost: true,
          firstRequestAdvertisedExec: true,
          execToolType: "custom",
          execFormatType: "grammar",
          execFormatSyntax: "lark",
          secondRequestHasCustomToolOutput: true,
          customToolOutputContainsMarker: true,
          requestErrors: [],
        },
        rendererSnapshot: {
          electron: true,
          hasInvokeBridge: true,
          supportsAppServer: true,
        },
        thread: { marker: "CODE_MODE_GATE_B_VISIBLE" },
        turn: { status: "completed", items: [{ type: "CodeCell" }] },
        visible: { finalAssistantTextVisible: true, toolRows: [] },
      }).publicCodeCellAbsent,
    ).toBe(false);
  });

  it("requires the standalone host to be an app-server child", () => {
    const evidence = readCodeModeProcessEvidence({
      electronPid: 100,
      appServerBinary: "/repo/target/debug/app-server",
      platform: "darwin",
      runner(command, args) {
        expect(command).toBe("ps");
        expect(args).toEqual(["-axo", "pid=,ppid=,command="]);
        return [
          "100 1 /Applications/Electron.app/Contents/MacOS/Electron",
          "101 100 /repo/target/debug/app-server --stdio",
          "102 101 /repo/target/debug/code-mode-host",
          "103 1 /repo/target/debug/code-mode-host",
        ].join("\n");
      },
    });

    expect(evidence).toMatchObject({
      electronPid: 100,
      appServerPid: 101,
      codeModeHostPid: 102,
      codeModeHostParentPid: 101,
    });
  });

  it("keeps runtime, provider capability, and Electron fixture boundaries explicit", () => {
    const source = fs.readFileSync(
      path.resolve("scripts/agent-runtime/code-mode-electron-gate-b.mjs"),
      "utf8",
    );
    const electronFixtureSource = fs.readFileSync(
      path.resolve("scripts/electron/mcp-config-fixture-smoke.mjs"),
      "utf8",
    );

    expect(source).toContain(
      'const OFFICIAL_OPENAI_BASE_URL = "http://api.openai.com/v1"',
    );
    expect(source).toContain('backendMode: "runtime"');
    expect(source).toContain("HTTP_PROXY");
    expect(source).toContain('modelToolMode: "code_mode"');
    expect(source).toContain('"custom_tools"');
    expect(source).toContain('path === "/v1/responses"');
    expect(source).toContain('call.method === "turn/start"');
    expect(source).toContain("codeModeHostOwnedByAppServer");
    expect(source).not.toContain('backendMode: "mock"');
    expect(source).not.toContain('type: "CodeCell"');
    expect(electronFixtureSource).toContain(
      'import { _electron as electron } from "playwright"',
    );
    expect(electronFixtureSource).toContain("electron.launch(");
  });
});
