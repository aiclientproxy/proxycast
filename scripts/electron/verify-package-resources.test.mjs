import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import {
  collectBareRuntimeImports,
  verifyElectronRuntimeBundles,
  verifyMacAppIdentity,
  verifyMacAppSignatures,
  verifyResourceRoot,
} from "./verify-package-resources.mjs";

const tmpRoots = [];

function createPackageRoot(infoPlistContent, helperInfoPlists = []) {
  const root = mkdtempSync(path.join(tmpdir(), "lime-electron-package-"));
  tmpRoots.push(root);
  const appContents = path.join(root, "mac-arm64", "Lime.app", "Contents");
  mkdirSync(appContents, { recursive: true });
  writeFileSync(path.join(appContents, "Info.plist"), infoPlistContent);
  for (const [helperName, helperInfoPlistContent] of helperInfoPlists) {
    const helperContents = path.join(
      appContents,
      "Frameworks",
      helperName,
      "Contents",
    );
    mkdirSync(helperContents, { recursive: true });
    writeFileSync(
      path.join(helperContents, "Info.plist"),
      helperInfoPlistContent,
    );
  }
  return root;
}

function createRuntimeBundleRoot({ main, preload }) {
  const root = mkdtempSync(path.join(tmpdir(), "lime-electron-runtime-"));
  tmpRoots.push(root);
  const mainDir = path.join(root, "dist-electron", "main");
  const preloadDir = path.join(root, "dist-electron", "preload");
  mkdirSync(mainDir, { recursive: true });
  mkdirSync(preloadDir, { recursive: true });
  writeFileSync(path.join(mainDir, "main.js"), main);
  writeFileSync(path.join(preloadDir, "preload.cjs"), preload);
  return root;
}

function sha256(content) {
  return createHash("sha256").update(content).digest("hex");
}

function createResourceRoot({
  appServer = "signed app-server",
  codeModeHost = "signed code-mode-host",
  manifestAppServer = "unsigned app-server",
  manifestCodeModeHost = "unsigned code-mode-host",
  platformKey = "darwin-arm64",
} = {}) {
  const root = mkdtempSync(path.join(tmpdir(), "lime-electron-resources-"));
  tmpRoots.push(root);
  const sidecarDir = path.join(root, "app-server", platformKey);
  const desktopAssetsDir = path.join(root, "desktop-assets");
  mkdirSync(sidecarDir, { recursive: true });
  mkdirSync(desktopAssetsDir, { recursive: true });
  const executableSuffix = platformKey.startsWith("win32-") ? ".exe" : "";
  writeFileSync(
    path.join(sidecarDir, `app-server${executableSuffix}`),
    appServer,
  );
  writeFileSync(
    path.join(sidecarDir, `code-mode-host${executableSuffix}`),
    codeModeHost,
  );
  for (const name of [
    "icon.png",
    "trayTemplate.png",
    "trayTemplate@2x.png",
    "tray-running.png",
    "tray-stopped.png",
    "tray-warning.png",
    "tray-error.png",
  ]) {
    writeFileSync(path.join(desktopAssetsDir, name), name);
  }
  writeFileSync(
    path.join(root, "app-server.release.json"),
    JSON.stringify({
      artifacts: [
        {
          platform: platformKey,
          sha256: sha256(manifestAppServer),
          codeModeHostSha256: sha256(manifestCodeModeHost),
        },
      ],
    }),
  );
  return root;
}

function buildInfoPlist(entries) {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<plist version="1.0">',
    "<dict>",
    ...entries.flatMap(([key, value]) => [
      `  <key>${key}</key>`,
      `  <string>${value}</string>`,
    ]),
    "</dict>",
    "</plist>",
  ].join("\n");
}

function cleanLimeInfoPlist(extraEntries = []) {
  return buildInfoPlist([
    ["CFBundleDisplayName", "Lime"],
    ["CFBundleName", "Lime"],
    ["CFBundleExecutable", "Lime"],
    ["CFBundleIdentifier", "com.limecloud.lime"],
    ["CFBundleIconFile", "icon.icns"],
    ...extraEntries,
  ]);
}

function cleanHelperInfoPlist(suffix = " (GPU)", extraEntries = []) {
  return buildInfoPlist([
    ["CFBundleDisplayName", `Lime Helper${suffix}`],
    ["CFBundleName", `Lime Helper${suffix}`],
    ["CFBundleExecutable", `Lime Helper${suffix}`],
    [
      "CFBundleIdentifier",
      `com.limecloud.lime.helper${suffix.replace(/[ ()]/g, "")}`,
    ],
    ...extraEntries,
  ]);
}

afterEach(() => {
  while (tmpRoots.length > 0) {
    const root = tmpRoots.pop();
    rmSync(root, { recursive: true, force: true });
  }
});

describe("verify-electron-package-resources runtime bundles", () => {
  it("收集运行时裸依赖包名", () => {
    expect(
      collectBareRuntimeImports(`
        import path from "node:path";
        import YAML from "yaml";
        import "@scope/pkg/register";
        const electron = require("electron");
        const local = require("./local.cjs");
      `),
    ).toEqual(["@scope/pkg", "electron", "yaml"]);
  });

  it("拒绝 Electron runtime bundle 残留非内置裸依赖", () => {
    const root = createRuntimeBundleRoot({
      main: 'import { parse } from "yaml";\nimport path from "node:path";\n',
      preload: 'const electron = require("electron");\n',
    });

    expect(() => verifyElectronRuntimeBundles(root)).toThrow(/yaml/);
  });

  it("接受只包含 node/electron/相对导入的 runtime bundle", () => {
    const root = createRuntimeBundleRoot({
      main: 'import path from "node:path";\nimport { Buffer } from "buffer";\nimport process from "process";\nimport "./chunk.js";\n',
      preload:
        'const electron = require("electron");\nconst local = require("./local.cjs");\n',
    });

    expect(() => verifyElectronRuntimeBundles(root)).not.toThrow();
  });
});

describe("verify-electron-package-resources sidecar integrity", () => {
  it("两个 sidecar 哈希一致时不调用 codesign", () => {
    const root = createResourceRoot({
      appServer: "app-server",
      codeModeHost: "code-mode-host",
      manifestAppServer: "app-server",
      manifestCodeModeHost: "code-mode-host",
    });

    const result = verifyResourceRoot(root, {
      platform: "darwin",
      arch: "arm64",
      execFileSyncImpl: () => {
        throw new Error("unexpected codesign call");
      },
    });

    expect(result.sha256.acceptedBecause).toBe("sha256");
    expect(result.codeModeHostIntegrity.acceptedBecause).toBe("sha256");
  });

  it("接受两个经过严格 codesign 验证的 macOS sidecar", () => {
    const root = createResourceRoot();
    const calls = [];

    const result = verifyResourceRoot(root, {
      platform: "darwin",
      arch: "arm64",
      execFileSyncImpl: (...args) => calls.push(args),
    });

    expect(result.sha256.acceptedBecause).toBe("macos-signed-sidecar");
    expect(result.codeModeHostIntegrity.acceptedBecause).toBe(
      "macos-signed-sidecar",
    );
    expect(calls).toEqual([
      [
        "codesign",
        ["--verify", "--strict", expect.stringMatching(/code-mode-host$/u)],
        { stdio: "ignore" },
      ],
      [
        "codesign",
        ["--verify", "--strict", expect.stringMatching(/app-server$/u)],
        { stdio: "ignore" },
      ],
    ]);
  });

  it("拒绝哈希变化且严格 codesign 失败的 macOS code-mode host", () => {
    const root = createResourceRoot();

    expect(() =>
      verifyResourceRoot(root, {
        platform: "darwin",
        arch: "arm64",
        execFileSyncImpl: () => {
          throw new Error("invalid signature");
        },
      }),
    ).toThrow(/code-mode host sidecar sha256 mismatch/u);
  });

  it("拒绝哈希变化且严格 codesign 失败的 macOS app-server", () => {
    const root = createResourceRoot({
      codeModeHost: "code-mode-host",
      manifestCodeModeHost: "code-mode-host",
    });

    expect(() =>
      verifyResourceRoot(root, {
        platform: "darwin",
        arch: "arm64",
        execFileSyncImpl: () => {
          throw new Error("invalid signature");
        },
      }),
    ).toThrow(/app-server sidecar sha256 mismatch/u);
  });

  it("非 macOS sidecar 哈希变化时不接受签名例外", () => {
    const root = createResourceRoot({ platformKey: "win32-x64" });
    let codesignCalled = false;

    expect(() =>
      verifyResourceRoot(root, {
        platform: "win32",
        arch: "x64",
        execFileSyncImpl: () => {
          codesignCalled = true;
        },
      }),
    ).toThrow(/code-mode host sidecar sha256 mismatch/u);
    expect(codesignCalled).toBe(false);
  });
});

describe("verify-electron-package-resources macOS app identity", () => {
  it("接受完整 Lime macOS app identity", () => {
    const root = createPackageRoot(cleanLimeInfoPlist());

    expect(verifyMacAppIdentity(root, { platform: "darwin" })).toEqual([
      expect.objectContaining({ kind: "main" }),
    ]);
  });

  it("接受 Lime helper app identity，不把 helper 名称误判成主 app", () => {
    const root = createPackageRoot(cleanLimeInfoPlist(), [
      ["Lime Helper (GPU).app", cleanHelperInfoPlist(" (GPU)")],
    ]);

    expect(verifyMacAppIdentity(root, { platform: "darwin" })).toEqual([
      expect.objectContaining({ kind: "main" }),
      expect.objectContaining({ kind: "helper" }),
    ]);
  });

  it("拒绝仍使用 Electron 可执行名的 macOS app", () => {
    const root = createPackageRoot(
      cleanLimeInfoPlist([["CFBundleExecutable", "Electron"]]),
    );

    expect(() => verifyMacAppIdentity(root, { platform: "darwin" })).toThrow(
      /executable still uses Electron/,
    );
  });

  it("拒绝 Forge packager extendInfo 字符串污染出的数字键", () => {
    const root = createPackageRoot(cleanLimeInfoPlist([["0", "l"]]));

    expect(() => verifyMacAppIdentity(root, { platform: "darwin" })).toThrow(
      /numeric extendInfo keys/,
    );
  });

  it("拒绝 helper app 中残留的 Electron Helper 品牌", () => {
    const root = createPackageRoot(cleanLimeInfoPlist(), [
      [
        "Lime Helper (GPU).app",
        cleanHelperInfoPlist(" (GPU)", [
          ["CFBundleName", "Electron Helper (GPU)"],
        ]),
      ],
    ]);

    expect(() => verifyMacAppIdentity(root, { platform: "darwin" })).toThrow(
      /helper app identity still uses Electron/,
    );
  });

  it("非 macOS 平台不检查 macOS app identity", () => {
    const root = createPackageRoot(
      cleanLimeInfoPlist([["CFBundleExecutable", "Electron"]]),
    );

    expect(verifyMacAppIdentity(root, { platform: "win32" })).toEqual([]);
  });
});

describe("verify-electron-package-resources macOS app signature", () => {
  it("用 deep strict codesign 验证主 app bundle", () => {
    const root = createPackageRoot(cleanLimeInfoPlist());
    const calls = [];

    expect(
      verifyMacAppSignatures(root, {
        platform: "darwin",
        execFileSyncImpl: (...args) => calls.push(args),
      }),
    ).toEqual([
      expect.objectContaining({
        valid: true,
        verification: "codesign --verify --deep --strict",
      }),
    ]);
    expect(calls).toEqual([
      [
        "codesign",
        ["--verify", "--deep", "--strict", expect.stringMatching(/Lime\.app$/)],
        { encoding: "utf8", stdio: "pipe" },
      ],
    ]);
  });

  it("拒绝没有 sealed resources 的无效 app 签名", () => {
    const root = createPackageRoot(cleanLimeInfoPlist());
    const failure = Object.assign(new Error("codesign failed"), {
      stderr: Buffer.from(
        "code has no resources but signature indicates they must be present",
      ),
    });

    expect(() =>
      verifyMacAppSignatures(root, {
        platform: "darwin",
        execFileSyncImpl: () => {
          throw failure;
        },
      }),
    ).toThrow(/code has no resources/);
  });

  it("非 macOS 平台不调用 codesign", () => {
    const root = createPackageRoot(cleanLimeInfoPlist());

    expect(
      verifyMacAppSignatures(root, {
        platform: "win32",
        execFileSyncImpl: () => {
          throw new Error("unexpected codesign call");
        },
      }),
    ).toEqual([]);
  });
});
