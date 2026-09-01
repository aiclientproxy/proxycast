import { createHash } from "node:crypto";
import {
  chmodSync,
  mkdtempSync,
  mkdirSync,
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
  NativeHostError,
  WindowsNativeHostClient,
  resolveWindowsNativeHostPath,
} from "./windowsNativeHost";

const originalPlatform = process.platform;
const originalResourcesPath = process.resourcesPath;
const roots: string[] = [];

function setPlatform(platform: NodeJS.Platform): void {
  Object.defineProperty(process, "platform", {
    configurable: true,
    value: platform,
  });
}

function setResourcesPath(value: string): void {
  Object.defineProperty(process, "resourcesPath", {
    configurable: true,
    value,
  });
}

function createFixture(content = "windows native host") {
  const root = mkdtempSync(path.join(tmpdir(), "lime-windows-native-host-"));
  roots.push(root);
  const helperPath = path.join(
    root,
    "native",
    "windows",
    "windows-native-host.exe",
  );
  mkdirSync(path.dirname(helperPath), { recursive: true });
  writeFileSync(helperPath, content);
  chmodSync(helperPath, 0o755);
  const sha256 = createHash("sha256").update(content).digest("hex");
  writeFileSync(
    path.join(root, "desktop-resources.manifest.json"),
    JSON.stringify({
      schemaVersion: 1,
      applicationId: "com.limecloud.lime",
      platform: "win32",
      arch: "x64",
      platformKey: "win32-x64",
      resources: [
        {
          id: "windows-native-host",
          kind: "helper",
          path: "native/windows/windows-native-host.exe",
          sha256,
          required: true,
        },
      ],
      native: {
        windowsHelper: {
          id: "windows-native-host",
          path: "native/windows/windows-native-host.exe",
          readOnly: true,
        },
      },
    }),
  );
  setResourcesPath(root);
  return { root, helperPath };
}

function createExecutableFixture(script: string) {
  return createFixture(`#!/usr/bin/env node\n${script}\n`);
}

beforeEach(() => {
  setPlatform("win32");
  appState.isPackaged = true;
});

afterEach(() => {
  setPlatform(originalPlatform);
  setResourcesPath(originalResourcesPath);
  appState.isPackaged = true;
  while (roots.length > 0)
    rmSync(roots.pop()!, { recursive: true, force: true });
});

describe("WindowsNativeHostClient", () => {
  it("匹配并发 JSONL 响应并转发 native event", async () => {
    const { helperPath } = createExecutableFixture(`
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", (line) => {
        const request = JSON.parse(line);
        const delay = request.params?.delay ?? 0;
        process.stdout.write(JSON.stringify({ event: "display.changed", payload: { id: request.id } }) + "\\n");
        setTimeout(() => {
          process.stdout.write(JSON.stringify({ id: request.id, ok: true, result: request.params.value }) + "\\n");
        }, delay);
      });
    `);
    const client = new WindowsNativeHostClient();
    const events: unknown[] = [];
    const unsubscribe = client.onEvent((event) => events.push(event));
    const first = client.invoke({
      method: "windows.displayWatcher.start",
      params: { value: "first", delay: 40 },
    });
    const second = client.invoke({
      method: "windows.displayWatcher.stop",
      params: { value: "second", delay: 0 },
    });
    await expect(Promise.all([first, second])).resolves.toEqual([
      "first",
      "second",
    ]);
    expect(events).toEqual([
      { event: "display.changed", payload: { id: "1" } },
      { event: "display.changed", payload: { id: "2" } },
    ]);
    expect(helperPath).toBeTruthy();
    unsubscribe();
    client.dispose();
  });

  it("helper 退出时拒绝所有 pending request", async () => {
    createExecutableFixture(`
      const readline = require("node:readline");
      const rl = readline.createInterface({ input: process.stdin });
      rl.on("line", () => process.exit(17));
    `);
    const client = new WindowsNativeHostClient();
    const first = client.invoke({ method: "windows.display.read" });
    const second = client.invoke({ method: "windows.window.read" });
    await expect(first).rejects.toThrow("Windows native host exited");
    await expect(second).rejects.toThrow("Windows native host exited");
    client.dispose();
  });

  it("请求超时后清理 pending 状态", async () => {
    vi.useFakeTimers();
    createExecutableFixture(`
      const readline = require("node:readline");
      readline.createInterface({ input: process.stdin });
      setInterval(() => {}, 1000);
    `);
    const client = new WindowsNativeHostClient();
    const pending = client.invoke({ method: "windows.displayWatcher.start" });
    const timeoutAssertion = expect(pending).rejects.toMatchObject({
      code: "timeout",
    });
    await vi.advanceTimersByTimeAsync(10_001);
    await timeoutAssertion;
    client.dispose();
    vi.useRealTimers();
  });

  it("仅解析身份、路径和 digest 均匹配的 helper", () => {
    const { helperPath } = createFixture();
    expect(resolveWindowsNativeHostPath()).toBe(helperPath);
    writeFileSync(helperPath, "tampered");
    expect(resolveWindowsNativeHostPath()).toBeNull();
  });

  it("manifest 缺失 helper metadata 时 fail closed", () => {
    const { root } = createFixture();
    writeFileSync(
      path.join(root, "desktop-resources.manifest.json"),
      JSON.stringify({
        schemaVersion: 1,
        applicationId: "com.limecloud.lime",
        platform: "win32",
        arch: "x64",
        platformKey: "win32-x64",
        resources: [],
      }),
    );
    expect(resolveWindowsNativeHostPath()).toBeNull();
  });

  it("非 Windows 平台拒绝调用", async () => {
    setPlatform("darwin");
    const client = new WindowsNativeHostClient();
    await expect(
      client.invoke({ method: "windows.uiAutomation.read" }),
    ).rejects.toBeInstanceOf(NativeHostError);
    await expect(
      client.invoke({ method: "windows.uiAutomation.read" }),
    ).rejects.toMatchObject({
      code: "unsupported",
    });
    client.dispose();
  });
});
