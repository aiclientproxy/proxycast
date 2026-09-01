import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readDesktopResourceReadiness } from "./desktopResourceReadiness";

const roots: string[] = [];

afterEach(() => {
  while (roots.length > 0) {
    rmSync(roots.pop()!, { recursive: true, force: true });
  }
});

function createResources(platform: "darwin" | "win32" = "darwin") {
  const root = mkdtempSync(path.join(tmpdir(), "lime-resource-readiness-"));
  roots.push(root);
  const key = platform === "darwin" ? "darwin-arm64" : "win32-x64";
  const suffix = platform === "win32" ? ".exe" : "";
  const names =
    platform === "darwin"
      ? ["app-server", "code-mode-host", "macos-native-host"]
      : [
          "app-server.exe",
          "code-mode-host.exe",
          "windows-sandbox-setup.exe",
          "windows-sandbox-runner.exe",
          "windows-native-host.exe",
        ];
  const resources = names.map((name) => {
    const relativePath =
      name === "macos-native-host"
        ? "native/macos/macos-native-host.app/Contents/MacOS/macos-native-host"
        : name === "windows-native-host.exe"
          ? "native/windows/windows-native-host.exe"
        : `app-server/${key}/${name}`;
    const absolutePath = path.join(root, relativePath);
    mkdirSync(path.dirname(absolutePath), { recursive: true });
    writeFileSync(absolutePath, name);
    return { id: name.replace(suffix, ""), path: relativePath, required: true };
  });
  writeFileSync(
    path.join(root, "desktop-resources.manifest.json"),
    JSON.stringify({
      schemaVersion: 1,
      applicationId: "com.limecloud.lime",
      platform,
      arch: platform === "darwin" ? "arm64" : "x64",
      platformKey: key,
      resources,
    }),
  );
  return root;
}

describe("desktop resource readiness", () => {
  it("识别 macOS 必需资源并保持签名/Gate B 未验证", () => {
    const root = createResources("darwin");
    expect(
      readDesktopResourceReadiness({
        resourcesRoot: root,
        platform: "darwin",
        arch: "arm64",
        packaged: true,
      }),
    ).toMatchObject({
      status: "unverified",
      platformKey: "darwin-arm64",
      resourceIds: ["app-server", "code-mode-host", "macos-native-host"],
    });
  });

  it("识别 Windows sidecar/sandbox 资源组", () => {
    const root = createResources("win32");
    expect(
      readDesktopResourceReadiness({
        resourcesRoot: root,
        platform: "win32",
        arch: "x64",
        packaged: true,
      }),
    ).toMatchObject({
      status: "unverified",
      platformKey: "win32-x64",
      resourceIds: [
        "app-server",
        "code-mode-host",
        "windows-sandbox-setup",
        "windows-sandbox-runner",
        "windows-native-host",
      ],
    });
  });

  it("资源缺失或 manifest 身份漂移时 fail closed", () => {
    const root = createResources("darwin");
    rmSync(
      path.join(
        root,
        "native/macos/macos-native-host.app/Contents/MacOS/macos-native-host",
      ),
    );
    expect(
      readDesktopResourceReadiness({
        resourcesRoot: root,
        platform: "darwin",
        arch: "arm64",
        packaged: true,
      }).status,
    ).toBe("unavailable");

    const identityRoot = createResources("darwin");
    const manifestPath = path.join(
      identityRoot,
      "desktop-resources.manifest.json",
    );
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    manifest.applicationId = "com.openai.codex";
    writeFileSync(manifestPath, JSON.stringify(manifest));
    expect(
      readDesktopResourceReadiness({
        resourcesRoot: identityRoot,
        platform: "darwin",
        arch: "arm64",
        packaged: true,
      }).status,
    ).toBe("unavailable");

    const windowsRoot = createResources("win32");
    rmSync(
      path.join(windowsRoot, "native/windows/windows-native-host.exe"),
    );
    expect(
      readDesktopResourceReadiness({
        resourcesRoot: windowsRoot,
        platform: "win32",
        arch: "x64",
        packaged: true,
      }),
    ).toMatchObject({
      status: "unavailable",
      reason: expect.stringContaining("windows-native-host"),
    });
  });
});
