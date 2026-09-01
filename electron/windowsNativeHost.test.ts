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
  const helperPath = path.join(root, "native", "windows", "windows-native-host.exe");
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

beforeEach(() => {
  setPlatform("win32");
  appState.isPackaged = true;
});

afterEach(() => {
  setPlatform(originalPlatform);
  setResourcesPath(originalResourcesPath);
  appState.isPackaged = true;
  while (roots.length > 0) rmSync(roots.pop()!, { recursive: true, force: true });
});

describe("WindowsNativeHostClient", () => {
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
    await expect(client.invoke({ method: "windows.uiAutomation.read" })).rejects.toBeInstanceOf(
      NativeHostError,
    );
    await expect(client.invoke({ method: "windows.uiAutomation.read" })).rejects.toMatchObject({
      code: "unsupported",
    });
    client.dispose();
  });
});
