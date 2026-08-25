import fs from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  STRICT_REVIEW_METHOD,
  STRICT_REVIEW_STARTED_AT_MS,
  assertStrictReviewEvidence,
  parseStrictReviewGateArgs,
  summarizeStrictReviewEvidence,
} from "./strict-review-gate-b.mjs";

function fixtureEvidence() {
  const threadId = "thread-strict";
  const turnId = "turn-strict";
  return summarizeStrictReviewEvidence({
    traceRaw: JSON.stringify([
      {
        command: "app_server_handle_json_lines",
        transport: "electron-ipc",
        status: "success",
        args_preview: {
          request: {
            lines: [
              JSON.stringify({
                jsonrpc: "2.0",
                id: 2,
                method: "turn/start",
                params: { threadId },
              }),
            ],
          },
        },
      },
      {
        command: "app_server_drain_events",
        transport: "electron-ipc",
        status: "success",
      },
    ]),
    errorRaw: "[]",
    dom: {
      statusVisible: true,
      inputbarVisible: true,
      protocolMethod: STRICT_REVIEW_METHOD,
      threadId,
      turnId,
      startedAtMs: STRICT_REVIEW_STARTED_AT_MS,
      titleVisible: true,
      descriptionVisible: true,
      nextStepVisible: true,
    },
    threadId,
    turnId,
    backendLedger: [
      {
        kind: "turnStart",
        threadId,
        turnId,
        eventTypes: ["guardian.review.started"],
      },
    ],
    setupRequests: [
      {
        command: "app_server_handle_json_lines",
        method: "thread/start",
        transport: "electron-ipc",
        status: "success",
      },
      {
        command: "app_server_handle_json_lines",
        method: "thread/resume",
        transport: "electron-ipc",
        status: "success",
      },
    ],
  });
}

describe("strict-review-gate-b", () => {
  it("accepts valid CLI options", () => {
    expect(
      parseStrictReviewGateArgs([
        "--prefix",
        "strict-review-fixture",
        "--timeout-ms",
        "90000",
        "--keep-temp",
      ]),
    ).toMatchObject({
      prefix: "strict-review-fixture",
      timeoutMs: 90_000,
      keepTemp: true,
    });
  });

  it("requires exact notification identity, Electron IPC and visible GUI", () => {
    const evidence = fixtureEvidence();
    expect(() => assertStrictReviewEvidence(evidence)).not.toThrow();
    expect(evidence.bridge.missingMethods).toEqual([]);
    expect(evidence.identity).toEqual({
      backendMatchesCanonicalTurn: true,
      domMatchesCanonicalTurn: true,
      exactStartedAt: true,
    });
  });

  it("fails when the GUI loses exact method identity", () => {
    const evidence = fixtureEvidence();
    evidence.gui.exactProtocolMethod = false;
    expect(() => assertStrictReviewEvidence(evidence)).toThrow(
      /exact Strict Review method/u,
    );
  });

  it("keeps production mock fallback out of the fixture", () => {
    const source = fs.readFileSync(
      path.resolve("scripts/electron/strict-review-gate-b.mjs"),
      "utf8",
    );
    expect(source).toContain('backendMode: "external"');
    expect(source).toContain("APP_SERVER_HANDLE_JSON_LINES_COMMAND");
    expect(source).toContain("app_server_drain_events");
    expect(source).toContain(STRICT_REVIEW_METHOD);
    expect(source).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(source).not.toContain("invokeMockOnly");
    expect(source).not.toContain("agent_runtime_");
  });
});
