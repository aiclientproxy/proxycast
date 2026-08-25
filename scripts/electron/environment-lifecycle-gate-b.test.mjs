import fs from "node:fs";
import { describe, expect, it } from "vitest";
import {
  ENVIRONMENT_LIFECYCLE_ID,
  ENVIRONMENT_LIFECYCLE_REQUIRED_METHODS,
  assertEnvironmentLifecycleEvidence,
  parseEnvironmentLifecycleGateArgs,
  summarizeEnvironmentLifecycleEvidence,
} from "./environment-lifecycle-gate-b.mjs";

const THREAD_ID = "thread-environment-gate-b";

function traceRaw(transport = "electron-ipc") {
  return JSON.stringify([
    ...[
      ["environment/status", { environmentId: ENVIRONMENT_LIFECYCLE_ID }],
      ["thread/read", { threadId: THREAD_ID }],
    ].map(([method, params]) => ({
      command: "app_server_handle_json_lines",
      transport,
      status: "success",
      args_preview: {
        request: {
          lines: [
            JSON.stringify({ jsonrpc: "2.0", id: method, method, params }),
          ],
        },
      },
    })),
    {
      command: "app_server_drain_events",
      transport,
      status: "success",
    },
  ]);
}

function stages() {
  const stage = (status, protocolMethod) => ({
    visible: true,
    environmentId: ENVIRONMENT_LIFECYCLE_ID,
    status,
    protocolMethod,
    triggerLifecycle: status,
  });
  return {
    connected: stage("connected", "thread/environment/connected"),
    disconnected: stage("disconnected", "thread/environment/disconnected"),
    reconnected: stage("connected", "thread/environment/connected"),
  };
}

function evidence(transport = "electron-ipc") {
  return summarizeEnvironmentLifecycleEvidence({
    traceRaw: traceRaw(transport),
    errorRaw: "[]",
    stages: stages(),
    threadId: THREAD_ID,
    setupRequests: [
      ...["environment/add", "thread/start"].map((method) => ({
        command: "app_server_handle_json_lines",
        method,
        transport,
        status: "success",
        params:
          method === "environment/add"
            ? { environmentId: ENVIRONMENT_LIFECYCLE_ID }
            : { threadId: THREAD_ID },
      })),
    ],
    remoteRequests: [1, 2].flatMap((connection) =>
      ["initialize", "environment/info", "environment/status"].map(
        (method) => ({ connection, method }),
      ),
    ),
  });
}

describe("Environment lifecycle Electron Gate B", () => {
  it("绑定 current bridge、Environment identity 和 GUI lifecycle", () => {
    const result = evidence();
    expect(() => assertEnvironmentLifecycleEvidence(result)).not.toThrow();
    expect(result.bridge.missingMethods).toEqual([]);
    expect(result.remote.connectionCount).toBe(2);
  });

  it("拒绝 mock transport 或缺失断线状态", () => {
    expect(() => assertEnvironmentLifecycleEvidence(evidence("mock"))).toThrow(
      /缺少 Environment current method|non-electron|非 electron-ipc/u,
    );
    const missingDisconnected = evidence();
    missingDisconnected.gui.disconnected.visible = false;
    expect(() =>
      assertEnvironmentLifecycleEvidence(missingDisconnected),
    ).toThrow(/disconnected/u);
  });

  it("校验 CLI 并禁止 retired/mock owner", () => {
    expect(
      parseEnvironmentLifecycleGateArgs(["--timeout-ms", "60000"], {
        evidenceDir: "/tmp/evidence",
        prefix: "environment-lifecycle",
        timeoutMs: 120_000,
        intervalMs: 200,
        keepTemp: false,
      }),
    ).toMatchObject({ timeoutMs: 60_000 });
    expect(() =>
      parseEnvironmentLifecycleGateArgs(["--prefix", "../unsafe"], {
        evidenceDir: "/tmp/evidence",
        prefix: "environment-lifecycle",
        timeoutMs: 120_000,
        intervalMs: 200,
        keepTemp: false,
      }),
    ).toThrow(/invalid evidence prefix/u);

    const source = fs.readFileSync(
      "scripts/electron/environment-lifecycle-gate-b.mjs",
      "utf8",
    );
    for (const method of ENVIRONMENT_LIFECYCLE_REQUIRED_METHODS) {
      expect(source).toContain(method);
    }
    expect(source).toContain("thread/environment/connected");
    expect(source).toContain("thread/environment/disconnected");
    expect(source).toContain("releaseReconnect");
    expect(source).toContain("ensureElectronFixtureBuild");
    expect(source).toContain('backendMode: "unavailable"');
    expect(source).not.toContain('APP_SERVER_BACKEND_MODE: "mock"');
    expect(source).not.toContain("invokeMockOnly");
    expect(source).not.toContain("agent_runtime_");
  });
});
