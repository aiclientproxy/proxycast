import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  APP_SERVER_PROTOCOL_VERSION,
  appServerResourceBinaryName,
  appServerResourcePlatformKey,
  buildElectronAppServerReleaseManifest,
  codeModeHostResourceBinaryName,
  copyElectronAppServerRuntimeLibraries,
  electronAppServerBinaryDestination,
  electronAppServerManifestPath,
  electronAppServerResourcesRoot,
  electronCodeModeHostBinaryDestination,
  prepareElectronAppServerAssets,
  resolveElectronAppServerRuntimeEnv,
  resolveElectronAppServerSherpaTargetTriple,
} from "./electron-app-server-assets.mjs";

describe("electron app-server assets", () => {
  it("按平台解析 resources sidecar 名称与平台 key", () => {
    expect(appServerResourceBinaryName("darwin")).toBe("app-server");
    expect(appServerResourceBinaryName("win32")).toBe("app-server.exe");
    expect(codeModeHostResourceBinaryName("darwin")).toBe("code-mode-host");
    expect(codeModeHostResourceBinaryName("win32")).toBe("code-mode-host.exe");
    expect(appServerResourcePlatformKey("darwin", "arm64")).toBe(
      "darwin-arm64",
    );
    expect(appServerResourcePlatformKey("darwin", "x64")).toBe("darwin-x64");
    expect(appServerResourcePlatformKey("win32", "x64")).toBe("win32-x64");
    expect(appServerResourcePlatformKey("linux", "x64")).toBe("linux-x64");
    expect(
      resolveElectronAppServerSherpaTargetTriple({
        platform: "darwin",
        arch: "arm64",
      }),
    ).toBe("aarch64-apple-darwin");
    expect(
      resolveElectronAppServerSherpaTargetTriple({
        platform: "win32",
        arch: "x64",
      }),
    ).toBe("x86_64-pc-windows-msvc");
    expect(
      resolveElectronAppServerSherpaTargetTriple({
        platform: "linux",
        arch: "x64",
      }),
    ).toBeNull();
  });

  it("解析 Electron resources 输出路径", () => {
    const outputRoot = electronAppServerResourcesRoot("/repo/lime");
    expect(outputRoot).toBe(path.resolve("/repo/lime/dist-electron"));
    expect(electronAppServerManifestPath({ outputRoot })).toBe(
      path.resolve("/repo/lime/dist-electron/app-server.release.json"),
    );
    expect(
      electronAppServerBinaryDestination({
        outputRoot,
        platform: "darwin",
        arch: "arm64",
      }),
    ).toBe(
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
    );
    expect(
      electronCodeModeHostBinaryDestination({
        outputRoot,
        platform: "darwin",
        arch: "arm64",
      }),
    ).toBe(
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      ),
    );
  });

  it("生成 Electron packaged app-server manifest", async () => {
    const manifest = await buildElectronAppServerReleaseManifest({
      binaryPath: "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      codeModeHostBinaryPath:
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      version: "1.59.0",
      platform: "darwin-arm64",
      sha256File: async () => "sha256",
    });

    expect(manifest).toEqual({
      version: "1.59.0",
      protocolVersion: APP_SERVER_PROTOCOL_VERSION,
      artifacts: [
        {
          platform: "darwin-arm64",
          url: "app-resource://app-server/darwin-arm64/app-server",
          sha256: "sha256",
          codeModeHostSha256: "sha256",
        },
      ],
    });
  });

  it("Windows packaged manifest 包含 sandbox setup 与 runner 哈希", async () => {
    const manifest = await buildElectronAppServerReleaseManifest({
      binaryPath:
        "/repo/lime/dist-electron/app-server/win32-x64/app-server.exe",
      codeModeHostBinaryPath:
        "/repo/lime/dist-electron/app-server/win32-x64/code-mode-host.exe",
      windowsSandboxSetupBinaryPath:
        "/repo/lime/dist-electron/app-server/win32-x64/windows-sandbox-setup.exe",
      windowsSandboxRunnerBinaryPath:
        "/repo/lime/dist-electron/app-server/win32-x64/windows-sandbox-runner.exe",
      version: "1.59.0",
      platform: "win32-x64",
      sha256File: async (filePath) => `sha256:${path.basename(filePath)}`,
    });

    expect(manifest.artifacts[0]).toEqual({
      platform: "win32-x64",
      url: "app-resource://app-server/win32-x64/app-server.exe",
      sha256: "sha256:app-server.exe",
      codeModeHostSha256: "sha256:code-mode-host.exe",
      windowsSandboxSetupSha256: "sha256:windows-sandbox-setup.exe",
      windowsSandboxRunnerSha256: "sha256:windows-sandbox-runner.exe",
    });
  });

  it("准备 sidecar 二进制和 release manifest", async () => {
    const calls = [];
    const outputRoot = path.resolve("/repo/lime/dist-electron");
    const result = await prepareElectronAppServerAssets({
      repoRoot: "/repo/lime",
      outputRoot,
      platform: "darwin",
      arch: "arm64",
      sourceBinary: "/repo/lime/lime-rs/target/debug/app-server",
      sourceCodeModeHostBinary:
        "/repo/lime/lime-rs/target/debug/code-mode-host",
      readPackageJson: async () => ({ version: "1.59.0" }),
      makeDir: async (...args) => calls.push(["mkdir", ...args]),
      copy: async (...args) => calls.push(["copy", ...args]),
      clearLaunchBlockingXattrs: async (...args) =>
        calls.push(["xattr", ...args]),
      getStat: async () => ({ mode: 0o755 }),
      changeMode: async (...args) => calls.push(["chmod", ...args]),
      prepareRuntimeBinary: (...args) => calls.push(["prepare", ...args]),
      copyRuntimeLibraries: async () => [],
      write: async (...args) => calls.push(["write", ...args]),
      sha256File: async (filePath) => {
        expect(calls.at(-1)).toEqual([
          "prepare",
          {
            binaryPath: path.resolve(
              "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
            ),
            platform: "darwin",
          },
        ]);
        expect([
          path.resolve(
            "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
          ),
          path.resolve(
            "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
          ),
        ]).toContain(filePath);
        return "sha256";
      },
    });

    expect(result.binaryPath).toBe(
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
    );
    expect(result.manifestPath).toBe(
      path.resolve("/repo/lime/dist-electron/app-server.release.json"),
    );
    expect(result.codeModeHostBinaryPath).toBe(
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      ),
    );
    expect(calls[0]).toEqual([
      "mkdir",
      path.resolve("/repo/lime/dist-electron/app-server/darwin-arm64"),
      { recursive: true },
    ]);
    expect(calls[1]).toEqual([
      "copy",
      path.resolve("/repo/lime/lime-rs/target/debug/app-server"),
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
    ]);
    expect(calls[2]).toEqual([
      "copy",
      path.resolve("/repo/lime/lime-rs/target/debug/code-mode-host"),
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      ),
    ]);
    expect(calls[3]).toEqual([
      "xattr",
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
      "darwin",
    ]);
    expect(calls[4]).toEqual([
      "xattr",
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      ),
      "darwin",
    ]);
    expect(calls[5]).toEqual([
      "chmod",
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
      0o755,
    ]);
    expect(calls[6]).toEqual([
      "chmod",
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
      ),
      0o755,
    ]);
    expect(calls[7]).toEqual([
      "prepare",
      {
        binaryPath: path.resolve(
          "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
        ),
        platform: "darwin",
      },
    ]);
    expect(calls[8]).toEqual([
      "prepare",
      {
        binaryPath: path.resolve(
          "/repo/lime/dist-electron/app-server/darwin-arm64/code-mode-host",
        ),
        platform: "darwin",
      },
    ]);
    expect(calls[9][0]).toBe("write");
    expect(calls[9][1]).toBe(
      path.resolve("/repo/lime/dist-electron/app-server.release.json"),
    );
  });

  it("准备 packaged resources 时忽略 APP_SERVER_BIN 作为复制源", async () => {
    const calls = [];
    const outputRoot = path.resolve("/repo/lime/dist-electron");
    const result = await prepareElectronAppServerAssets({
      repoRoot: "/repo/lime",
      outputRoot,
      platform: "darwin",
      arch: "arm64",
      env: {
        APP_SERVER_BIN:
          "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
        CARGO_TARGET_DIR: "/repo/lime/lime-rs/target",
      },
      resolveBinary: (options) => {
        expect(options.env.APP_SERVER_BIN).toBeUndefined();
        expect(options.env.CARGO_TARGET_DIR).toBe("/repo/lime/lime-rs/target");
        return "/repo/lime/lime-rs/target/debug/app-server";
      },
      readPackageJson: async () => ({ version: "1.59.0" }),
      makeDir: async (...args) => calls.push(["mkdir", ...args]),
      copy: async (...args) => calls.push(["copy", ...args]),
      clearLaunchBlockingXattrs: async (...args) =>
        calls.push(["xattr", ...args]),
      getStat: async () => ({ mode: 0o755 }),
      changeMode: async (...args) => calls.push(["chmod", ...args]),
      prepareRuntimeBinary: async () => undefined,
      copyRuntimeLibraries: async () => [],
      write: async (...args) => calls.push(["write", ...args]),
      sha256File: async () => "sha256",
    });

    expect(result.sourceBinary).toBe(
      path.resolve("/repo/lime/lime-rs/target/debug/app-server"),
    );
    expect(calls[1]).toEqual([
      "copy",
      path.resolve("/repo/lime/lime-rs/target/debug/app-server"),
      path.resolve(
        "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
      ),
    ]);
  });

  it("准备 sidecar resources 时复制 sherpa/onnxruntime 运行库到同目录", async () => {
    const calls = [];
    const copied = await copyElectronAppServerRuntimeLibraries({
      repoRoot: "/repo/lime",
      platform: "darwin",
      arch: "arm64",
      sourceBinary: "/repo/lime/custom-target/debug/app-server",
      destinationDirectory: "/repo/lime/dist-electron/app-server/darwin-arm64",
      readCargoLock: async (filePath, encoding) => {
        expect(filePath).toBe("/repo/lime/lime-rs/Cargo.lock");
        expect(encoding).toBe("utf8");
        return `
[[package]]
name = "sherpa-onnx-sys"
version = "1.13.0"
`;
      },
      makeDir: async (...args) => calls.push(["mkdir", ...args]),
      copy: async (...args) => calls.push(["copy", ...args]),
      exists: (filePath) =>
        filePath ===
        "/repo/lime/custom-target/debug/libsherpa-onnx-c-api.dylib",
      resolvePlan: ({ repoRoot, targetTriple, version }) => {
        expect(repoRoot).toBe("/repo/lime");
        expect(targetTriple).toBe("aarch64-apple-darwin");
        expect(version).toBe("1.13.0");
        return {
          targetTriple,
          libs: ["libonnxruntime.1.24.4.dylib", "libsherpa-onnx-c-api.dylib"],
        };
      },
      resolveLibrary: (_plan, name) => `/repo/lime/prebuilt/${name}`,
    });

    expect(copied).toEqual([
      {
        name: "libonnxruntime.1.24.4.dylib",
        sourcePath: "/repo/lime/prebuilt/libonnxruntime.1.24.4.dylib",
        required: true,
        destinationPath:
          "/repo/lime/dist-electron/app-server/darwin-arm64/libonnxruntime.1.24.4.dylib",
      },
      {
        name: "libsherpa-onnx-c-api.dylib",
        sourcePath: "/repo/lime/custom-target/debug/libsherpa-onnx-c-api.dylib",
        required: true,
        destinationPath:
          "/repo/lime/dist-electron/app-server/darwin-arm64/libsherpa-onnx-c-api.dylib",
      },
      {
        name: "libsherpa-onnx-cxx-api.dylib",
        sourcePath: "/repo/lime/prebuilt/libsherpa-onnx-cxx-api.dylib",
        required: false,
        destinationPath:
          "/repo/lime/dist-electron/app-server/darwin-arm64/libsherpa-onnx-cxx-api.dylib",
      },
    ]);
    expect(calls).toEqual([
      [
        "mkdir",
        "/repo/lime/dist-electron/app-server/darwin-arm64",
        { recursive: true },
      ],
      [
        "copy",
        "/repo/lime/prebuilt/libonnxruntime.1.24.4.dylib",
        "/repo/lime/dist-electron/app-server/darwin-arm64/libonnxruntime.1.24.4.dylib",
      ],
      [
        "copy",
        "/repo/lime/custom-target/debug/libsherpa-onnx-c-api.dylib",
        "/repo/lime/dist-electron/app-server/darwin-arm64/libsherpa-onnx-c-api.dylib",
      ],
      [
        "copy",
        "/repo/lime/prebuilt/libsherpa-onnx-cxx-api.dylib",
        "/repo/lime/dist-electron/app-server/darwin-arm64/libsherpa-onnx-cxx-api.dylib",
      ],
    ]);
  });

  it("缺少必需运行库时 fail fast", async () => {
    await expect(
      copyElectronAppServerRuntimeLibraries({
        repoRoot: "/repo/lime",
        platform: "darwin",
        arch: "arm64",
        sourceBinary: "/repo/lime/custom-target/debug/app-server",
        destinationDirectory:
          "/repo/lime/dist-electron/app-server/darwin-arm64",
        readCargoLock: async () => `
[[package]]
name = "sherpa-onnx-sys"
version = "1.13.0"
`,
        exists: () => false,
        resolvePlan: ({ targetTriple }) => ({
          targetTriple,
          libs: ["libsherpa-onnx-c-api.dylib"],
        }),
        resolveLibrary: () => null,
      }),
    ).rejects.toThrow(
      "Expected app-server runtime library missing for aarch64-apple-darwin: libsherpa-onnx-c-api.dylib",
    );
  });

  it("拒绝把 packaged destination 当作 sourceBinary", async () => {
    const destination = path.resolve(
      "/repo/lime/dist-electron/app-server/darwin-arm64/app-server",
    );

    await expect(
      prepareElectronAppServerAssets({
        repoRoot: "/repo/lime",
        outputRoot: path.resolve("/repo/lime/dist-electron"),
        platform: "darwin",
        arch: "arm64",
        sourceBinary: destination,
        readPackageJson: async () => ({ version: "1.59.0" }),
      }),
    ).rejects.toThrow(
      "Electron app-server asset source must not equal packaged destination",
    );
  });

  it("runtime env 优先使用显式 APP_SERVER_BIN", () => {
    const prepared = [];
    expect(
      resolveElectronAppServerRuntimeEnv({
        env: { APP_SERVER_BIN: "  /custom/app-server  " },
        exists: () => true,
        resolveBinary: () => {
          throw new Error("should not resolve");
        },
        prepareRuntimeBinary: (...args) => prepared.push(args),
      }),
    ).toEqual({ APP_SERVER_BIN: "/custom/app-server" });
    expect(prepared).toEqual([
      [{ binaryPath: "/custom/app-server", platform: process.platform }],
    ]);
  });

  it("runtime env 已存在 manifest 时不注入开发 sidecar", () => {
    expect(
      resolveElectronAppServerRuntimeEnv({
        env: {},
        repoRoot: "/repo/lime",
        exists: () => true,
        resolveBinary: () => {
          throw new Error("should not resolve");
        },
      }),
    ).toEqual({});
  });

  it("runtime env 无 manifest 时回退开发 sidecar", () => {
    const prepared = [];
    expect(
      resolveElectronAppServerRuntimeEnv({
        env: {},
        repoRoot: "/repo/lime",
        platform: "darwin",
        exists: () => false,
        resolveBinary: (options) => {
          expect(options.repoRoot).toBe("/repo/lime");
          expect(options.platform).toBe("darwin");
          return "/repo/lime/lime-rs/target/debug/app-server";
        },
        prepareRuntimeBinary: (...args) => prepared.push(args),
      }),
    ).toEqual({ APP_SERVER_BIN: "/repo/lime/lime-rs/target/debug/app-server" });
    expect(prepared).toEqual([
      [
        {
          binaryPath: "/repo/lime/lime-rs/target/debug/app-server",
          platform: "darwin",
        },
      ],
    ]);
  });
});
