import { createHash } from "node:crypto";
import {
  chmodSync,
  mkdtempSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { appState } = vi.hoisted(() => ({
  appState: { isPackaged: true },
}));

vi.mock("./electronRuntime", () => ({
  app: {
    get isPackaged() {
      return appState.isPackaged;
    },
  },
}));

import {
  MacOSNativeHostClient,
  NativeHostError,
  resolveMacOSNativeHostPath,
} from "./macosNativeHost";

const originalPlatform = process.platform;
const originalResourcesPath = process.resourcesPath;
const tempRoots: string[] = [];

function setPlatform(platform: NodeJS.Platform): void {
  Object.defineProperty(process, "platform", {
    configurable: true,
    value: platform,
  });
}

function setResourcesPath(resourcesPath: string): void {
  Object.defineProperty(process, "resourcesPath", {
    configurable: true,
    value: resourcesPath,
  });
}

function createNativeHostFixture(
  script: string,
  protocolVersion?: number,
): string {
  const root = mkdtempSync(path.join(tmpdir(), "lime-macos-native-host-"));
  tempRoots.push(root);
  const helperPath = path.join(root, "native", "macos-native-host");
  const helperDir = path.dirname(helperPath);
  const content = `#!/usr/bin/env node\n${script}\n`;
  mkdirSync(helperDir, { recursive: true });
  writeFileSync(helperPath, content);
  chmodSync(helperPath, 0o755);
  const digest = createHash("sha256")
    .update(readFileSync(helperPath))
    .digest("hex");
  writeFileSync(
    path.join(root, "desktop-resources.manifest.json"),
    JSON.stringify({
      schemaVersion: 1,
      applicationId: "com.limecloud.lime",
      platform: "darwin",
      arch: process.arch,
      platformKey: `darwin-${process.arch === "arm64" ? "arm64" : "x64"}`,
      resources: [
        {
          id: "macos-native-host",
          kind: "helper",
          path: "native/macos-native-host",
          sha256: digest,
          required: true,
        },
      ],
      native: {
        helper: {
          id: "macos-native-host",
          path: "native/macos-native-host",
          ...(protocolVersion === undefined ? {} : { protocolVersion }),
          bundleIdentifier: "com.limecloud.lime.native-host",
          signedByForge: false,
        },
      },
    }),
  );
  return root;
}

function createClient(
  script: string,
  protocolVersion?: number,
): {
  client: MacOSNativeHostClient;
  root: string;
} {
  const root = createNativeHostFixture(script, protocolVersion);
  setResourcesPath(root);
  return { client: new MacOSNativeHostClient(), root };
}

beforeEach(() => {
  setPlatform("darwin");
  appState.isPackaged = true;
});

afterEach(() => {
  setPlatform(originalPlatform);
  setResourcesPath(originalResourcesPath);
  appState.isPackaged = true;
  while (tempRoots.length > 0) {
    rmSync(tempRoots.pop()!, { recursive: true, force: true });
  }
});

describe("MacOSNativeHostClient", () => {
  it("匹配并发 JSONL 响应且允许响应乱序返回", async () => {
    const { client } = createClient(`
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", (line) => {
        const request = JSON.parse(line);
        const delay = request.params?.delay ?? 0;
        process.stdout.write(JSON.stringify({ event: "native.progress", payload: { id: request.id } }) + "\\n");
        setTimeout(() => {
          process.stdout.write(JSON.stringify({ id: request.id, ok: true, result: request.params.value }) + "\\n");
        }, delay);
      });
    `);

    const events: unknown[] = [];
    const unsubscribe = client.onEvent((event) => events.push(event));
    const first = client.invoke({
      method: "test.echo",
      params: { value: "first", delay: 40 },
    });
    const second = client.invoke({
      method: "test.echo",
      params: { value: "second", delay: 0 },
    });
    await expect(Promise.all([first, second])).resolves.toEqual([
      "first",
      "second",
    ]);
    expect(events).toEqual([
      { event: "native.progress", payload: { id: "1" } },
      { event: "native.progress", payload: { id: "2" } },
    ]);
    unsubscribe();
    client.dispose();
  });

  it("helper 缺失时返回 unavailable 且不启动子进程", async () => {
    const { client, root } = createClient("process.stdin.resume();");
    rmSync(path.join(root, "native", "macos-native-host"));

    await expect(client.invoke({ method: "test.echo" })).rejects.toMatchObject({
      code: "unavailable",
    });
    expect(resolveMacOSNativeHostPath()).toBeNull();
    client.dispose();
  });

  it("manifest 哈希漂移时 fail closed", async () => {
    const { client, root } = createClient("process.stdin.resume();");
    writeFileSync(path.join(root, "native", "macos-native-host"), "tampered");

    await expect(client.invoke({ method: "test.echo" })).rejects.toMatchObject({
      code: "unavailable",
    });
    client.dispose();
  });

  it("helper 退出时拒绝所有 pending request", async () => {
    const { client } = createClient(`
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", () => process.exit(17));
    `);

    const first = client.invoke({ method: "test.exit" });
    const second = client.invoke({ method: "test.exit" });
    await expect(first).rejects.toThrow("macOS native host exited");
    await expect(second).rejects.toThrow("macOS native host exited");
    client.dispose();
  });

  it("打包 helper 声明协议版本时先完成身份握手且只握手一次", async () => {
    const { client } = createClient(
      `
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      let handshakes = 0;
      rl.on("line", (line) => {
        const request = JSON.parse(line);
        if (request.method === "capabilities.read") {
          handshakes += 1;
          process.stdout.write(JSON.stringify({
            id: request.id,
            ok: true,
            result: {
              protocolVersion: 1,
              helperId: "macos-native-host",
              platform: "darwin",
              applicationId: "com.limecloud.lime.native-host",
            },
          }) + "\\n");
          return;
        }
        process.stdout.write(JSON.stringify({
          id: request.id,
          ok: true,
          result: { method: request.method, handshakes },
        }) + "\\n");
      });
    `,
      1,
    );

    await expect(client.invoke({ method: "test.echo" })).resolves.toEqual({
      method: "test.echo",
      handshakes: 1,
    });
    await expect(client.invoke({ method: "test.echo" })).resolves.toEqual({
      method: "test.echo",
      handshakes: 1,
    });
    client.dispose();
  });

  it("打包 helper 握手身份漂移时 fail closed", async () => {
    const { client } = createClient(
      `
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", (line) => {
        const request = JSON.parse(line);
        process.stdout.write(JSON.stringify({
          id: request.id,
          ok: true,
          result: {
            protocolVersion: 99,
            helperId: "foreign-helper",
            platform: "darwin",
            applicationId: "com.openai.codex.native-host",
          },
        }) + "\\n");
      });
    `,
      1,
    );

    await expect(client.invoke({ method: "test.echo" })).rejects.toMatchObject({
      code: "protocol_mismatch",
    });
    client.dispose();
  });

  it("请求超时后清理 pending 状态", async () => {
    vi.useFakeTimers();
    const { client } = createClient(`
      const readline = require("node:readline");
      readline.createInterface({ input: process.stdin });
      setInterval(() => {}, 1000);
    `);

    const pending = client.invoke({ method: "test.hang" });
    const timeoutAssertion = expect(pending).rejects.toMatchObject({
      code: "timeout",
    });
    await vi.advanceTimersByTimeAsync(10_001);
    await timeoutAssertion;
    client.dispose();
    vi.useRealTimers();
  });

  it("通过 JSONL 边界传递 Apple Events 目标和授权查询方法", async () => {
    const { client } = createClient(`
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", (line) => {
        const request = JSON.parse(line);
        if (request.method === "appleEvents.targets") {
          process.stdout.write(JSON.stringify({
            id: request.id,
            ok: true,
            result: { targets: [{ bundleId: "com.apple.finder" }] },
          }) + "\\n");
          return;
        }
        process.stdout.write(JSON.stringify({
          id: request.id,
          ok: true,
          result: {
            method: request.method,
            targetBundleId: request.params.targetBundleId,
            askedUser: request.method === "appleEvents.request",
          },
        }) + "\\n");
      });
    `);

    await expect(
      client.invoke({
        method: "appleEvents.read",
        params: { targetBundleId: "com.apple.finder" },
      }),
    ).resolves.toEqual({
      method: "appleEvents.read",
      targetBundleId: "com.apple.finder",
      askedUser: false,
    });
    await expect(
      client.invoke({
        method: "appleEvents.request",
        params: { targetBundleId: "com.apple.finder" },
      }),
    ).resolves.toEqual({
      method: "appleEvents.request",
      targetBundleId: "com.apple.finder",
      askedUser: true,
    });
    await expect(
      client.invoke({ method: "appleEvents.targets" }),
    ).resolves.toEqual({ targets: [{ bundleId: "com.apple.finder" }] });
    await expect(
      client.invoke({ method: "appleEvents.openSettings" }),
    ).resolves.toEqual({
      method: "appleEvents.openSettings",
      askedUser: false,
    });
    client.dispose();
  });

  it("非 macOS 平台 fail closed", async () => {
    setPlatform("win32");
    const client = new MacOSNativeHostClient();

    await expect(client.invoke({ method: "test.echo" })).rejects.toEqual(
      expect.objectContaining<Partial<NativeHostError>>({
        code: "unsupported",
      }),
    );
    expect(resolveMacOSNativeHostPath()).toBeNull();
    client.dispose();
  });
});
