import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

import { buildElectronUpdateFeedUploadPlan } from "./update-feed-r2-upload-plan.mjs";
import { planR2ReleaseCleanup } from "./r2-release-cleanup-plan.mjs";
import { prepareGitHubReleaseAssets } from "./prepare-github-release-assets.mjs";
import { stageElectronReleaseAssets } from "./stage-release-assets.mjs";

function writeFile(filePath, content = "asset") {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}

describe("Electron updater upload plan", () => {
  it("Forge / Squirrel updater 资产应写入 feed 与版本化 R2 路径", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-update-feed-plan-"),
    );
    const assetsDir = path.join(root, "release-assets");

    writeFile(
      path.join(assetsDir, "aarch64-apple-darwin", "Lime.dmg"),
      "arm-installer",
    );
    writeFile(
      path.join(
        assetsDir,
        "aarch64-apple-darwin",
        "Lime-darwin-arm64-1.20.0.zip",
      ),
      "arm-zip",
    );
    writeFile(
      path.join(assetsDir, "aarch64-apple-darwin", "RELEASES.json"),
      "arm-feed",
    );
    writeFile(
      path.join(assetsDir, "x86_64-apple-darwin", "Lime.dmg"),
      "x64-installer",
    );
    writeFile(
      path.join(assetsDir, "x86_64-apple-darwin", "Lime-darwin-x64-1.20.0.zip"),
      "x64-zip",
    );
    writeFile(
      path.join(assetsDir, "x86_64-apple-darwin", "RELEASES.json"),
      "x64-feed",
    );
    writeFile(
      path.join(assetsDir, "x86_64-pc-windows-msvc", "Lime-1.20.0 Setup.exe"),
      "win-installer",
    );
    writeFile(
      path.join(assetsDir, "x86_64-pc-windows-msvc", "lime-1.20.0-full.nupkg"),
      "win-package",
    );
    writeFile(
      path.join(assetsDir, "x86_64-pc-windows-msvc", "RELEASES"),
      "win-feed",
    );

    const plan = buildElectronUpdateFeedUploadPlan({
      assetsDir,
      channel: "stable",
      version: "v1.20.0",
    });

    expect(plan.map((item) => item.key)).toEqual([
      "lime/stable/darwin-arm64/Lime-darwin-arm64-1.20.0.zip",
      "lime/stable/darwin-arm64/Lime.dmg",
      "lime/stable/darwin-arm64/RELEASES.json",
      "lime/stable/darwin-x64/Lime-darwin-x64-1.20.0.zip",
      "lime/stable/darwin-x64/Lime.dmg",
      "lime/stable/darwin-x64/RELEASES.json",
      "lime/stable/v1.20.0/darwin-arm64/Lime-darwin-arm64-1.20.0.zip",
      "lime/stable/v1.20.0/darwin-arm64/Lime.dmg",
      "lime/stable/v1.20.0/darwin-arm64/RELEASES.json",
      "lime/stable/v1.20.0/darwin-x64/Lime-darwin-x64-1.20.0.zip",
      "lime/stable/v1.20.0/darwin-x64/Lime.dmg",
      "lime/stable/v1.20.0/darwin-x64/RELEASES.json",
      "lime/stable/v1.20.0/win32-x64/Lime-1.20.0 Setup.exe",
      "lime/stable/v1.20.0/win32-x64/lime-1.20.0-full.nupkg",
      "lime/stable/v1.20.0/win32-x64/RELEASES",
      "lime/stable/win32-x64/Lime-1.20.0 Setup.exe",
      "lime/stable/win32-x64/lime-1.20.0-full.nupkg",
      "lime/stable/win32-x64/RELEASES",
    ]);
    expect(
      plan
        .filter((item) =>
          /^RELEASES(?:\.json)?$/.test(path.basename(item.file)),
        )
        .every((item) => item.cacheControl.includes("max-age=60")),
    ).toBe(true);
    expect(
      plan.find((item) => path.basename(item.file) === "RELEASES.json")
        ?.contentType,
    ).toBe("application/json");
    expect(
      plan.find((item) => path.basename(item.file) === "RELEASES")?.contentType,
    ).toBe("text/plain");
  });

  it("拒绝旧 updater 资产进入 Electron R2 上传计划", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-update-feed-legacy-"),
    );
    writeFile(
      path.join(
        root,
        "release-assets",
        "aarch64-apple-darwin",
        "Lime.app.tar.gz",
      ),
    );

    expect(() =>
      buildElectronUpdateFeedUploadPlan({
        assetsDir: path.join(root, "release-assets"),
        version: "v1.20.0",
      }),
    ).toThrow(/legacy updater assets/);
  });
});

describe("Electron release asset staging", () => {
  it("拒绝旧 updater 资产停留在 Electron Forge 输出目录", async () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-legacy-"),
    );
    const builderDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "aarch64-apple-darwin");

    writeFile(path.join(builderDir, "Lime_1.20.0_aarch64.dmg"));
    writeFile(path.join(builderDir, "latest-mac.yml"));
    writeFile(path.join(builderDir, "Lime.app.tar.gz"));

    expect(() =>
      stageElectronReleaseAssets({
        forgeDir: builderDir,
        outDir,
        targetTriple: "aarch64-apple-darwin",
        version: "v1.20.0",
      }),
    ).toThrow(/legacy updater assets are not allowed/);
  });

  it("macOS Forge 输出应复制 RELEASES.json、DMG 和 zip", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-forge-mac-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "aarch64-apple-darwin");

    writeFile(path.join(forgeDir, "make", "Lime.dmg"), "arm-installer");
    writeFile(
      path.join(
        forgeDir,
        "make",
        "zip",
        "darwin",
        "arm64",
        "Lime-darwin-arm64-1.20.0.zip",
      ),
      "arm-zip",
    );
    writeFile(
      path.join(forgeDir, "make", "zip", "darwin", "arm64", "RELEASES.json"),
      JSON.stringify({ currentRelease: "1.20.0" }),
    );

    const copied = stageElectronReleaseAssets({
      forgeDir,
      outDir,
      targetTriple: "aarch64-apple-darwin",
      version: "v1.20.0",
    });

    expect(
      copied.map((item) => path.basename(item.destination)).sort(),
    ).toEqual(
      ["Lime-darwin-arm64-1.20.0.zip", "Lime.dmg", "RELEASES.json"].sort(),
    );
    expect(
      fs.readFileSync(path.join(outDir, "RELEASES.json"), "utf8"),
    ).toContain("1.20.0");
  });

  it("macOS staging 应拒绝本地临时 feed 生成的 RELEASES.json", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-local-feed-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "aarch64-apple-darwin");

    writeFile(path.join(forgeDir, "make", "Lime.dmg"), "arm-installer");
    writeFile(
      path.join(
        forgeDir,
        "make",
        "zip",
        "darwin",
        "arm64",
        "Lime-darwin-arm64-1.20.0.zip",
      ),
      "arm-zip",
    );
    writeFile(
      path.join(forgeDir, "make", "zip", "darwin", "arm64", "RELEASES.json"),
      JSON.stringify({
        currentRelease: "1.20.0",
        releases: [
          {
            version: "1.20.0",
            updateTo: {
              url: "http://127.0.0.1:12345/lime/stable/darwin-arm64/Lime-darwin-arm64-1.20.0.zip",
            },
          },
        ],
      }),
    );

    expect(() =>
      stageElectronReleaseAssets({
        forgeDir,
        outDir,
        targetTriple: "aarch64-apple-darwin",
        version: "v1.20.0",
      }),
    ).toThrow(/local Electron updater feed URLs are not allowed/);
  });

  it("Windows Forge 输出应复制 Squirrel RELEASES、nupkg 和 Setup", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-forge-win-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "x86_64-pc-windows-msvc");

    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "Lime-1.20.0 Setup.exe",
      ),
      "setup",
    );
    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "lime-1.20.0-full.nupkg",
      ),
      "nupkg",
    );
    writeFile(
      path.join(forgeDir, "make", "squirrel.windows", "x64", "RELEASES"),
      "releases",
    );

    const copied = stageElectronReleaseAssets({
      forgeDir,
      outDir,
      targetTriple: "x86_64-pc-windows-msvc",
      version: "v1.20.0",
    });

    expect(
      copied.map((item) => path.basename(item.destination)).sort(),
    ).toEqual(
      ["Lime-1.20.0 Setup.exe", "RELEASES", "lime-1.20.0-full.nupkg"].sort(),
    );
  });

  it("Windows staging 不应把 packaged windows-sandbox sidecar 当成 Squirrel 安装包", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-forge-win-sidecar-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "x86_64-pc-windows-msvc");

    writeFile(
      path.join(
        forgeDir,
        "Lime-win32-x64",
        "resources",
        "app-server",
        "win32-x64",
        "windows-sandbox-setup.exe",
      ),
      "sidecar",
    );
    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "Lime-1.20.0 Setup.exe",
      ),
      "installer",
    );
    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "lime-1.20.0-full.nupkg",
      ),
      "nupkg",
    );
    writeFile(
      path.join(forgeDir, "make", "squirrel.windows", "x64", "RELEASES"),
      "releases",
    );

    const copied = stageElectronReleaseAssets({
      forgeDir,
      outDir,
      targetTriple: "x86_64-pc-windows-msvc",
      version: "v1.20.0",
    });

    expect(
      copied.map((item) => path.basename(item.destination)).sort(),
    ).toEqual(
      ["Lime-1.20.0 Setup.exe", "RELEASES", "lime-1.20.0-full.nupkg"].sort(),
    );
    expect(
      fs.readFileSync(path.join(outDir, "Lime-1.20.0 Setup.exe"), "utf8"),
    ).toBe("installer");
    expect(fs.existsSync(path.join(outDir, "windows-sandbox-setup.exe"))).toBe(
      false,
    );
  });

  it("Windows staging 缺少 Squirrel nupkg 时应输出 Forge 候选资产", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-forge-win-missing-nupkg-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "x86_64-pc-windows-msvc");

    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "Lime-1.20.0 Setup.exe",
      ),
      "setup",
    );
    writeFile(
      path.join(forgeDir, "make", "squirrel.windows", "x64", "RELEASES"),
      "releases",
    );

    expect(() =>
      stageElectronReleaseAssets({
        forgeDir,
        outDir,
        targetTriple: "x86_64-pc-windows-msvc",
        version: "v1.20.0",
      }),
    ).toThrow(
      /no updater archive for x86_64-pc-windows-msvc asset found under Forge output\. Candidate files: .*Lime-1\.20\.0 Setup\.exe.*RELEASES/,
    );
  });

  it("Windows staging 不应把非 Setup exe 当作 Squirrel 安装包", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-electron-stage-forge-win-non-setup-exe-"),
    );
    const forgeDir = path.join(root, "release-electron");
    const outDir = path.join(root, "release-assets", "x86_64-pc-windows-msvc");

    writeFile(
      path.join(forgeDir, "make", "squirrel.windows", "x64", "Lime.exe"),
      "app-exe",
    );
    writeFile(
      path.join(
        forgeDir,
        "make",
        "squirrel.windows",
        "x64",
        "lime-1.20.0-full.nupkg",
      ),
      "nupkg",
    );
    writeFile(
      path.join(forgeDir, "make", "squirrel.windows", "x64", "RELEASES"),
      "releases",
    );

    expect(() =>
      stageElectronReleaseAssets({
        forgeDir,
        outDir,
        targetTriple: "x86_64-pc-windows-msvc",
        version: "v1.20.0",
      }),
    ).toThrow(/no installer for x86_64-pc-windows-msvc asset found/);
  });
});

describe("R2 release cleanup", () => {
  it("只删除超过保留窗口且未受保护的旧版本", () => {
    const keys = [
      "lime/stable/v1.20.0/win32-x64/RELEASES",
      "lime/stable/v1.20.0/win32-x64/Lime-1.20.0 Setup.exe",
      "lime/stable/v1.19.0/win32-x64/Lime-1.19.0 Setup.exe",
      "lime/stable/v1.18.0/win32-x64/Lime-1.18.0 Setup.exe",
      "lime/stable/v1.17.0/win32-x64/Lime-1.17.0 Setup.exe",
      "lime/stable/v1.16.0/win32-x64/Lime-1.16.0 Setup.exe",
    ];

    const plan = planR2ReleaseCleanup({
      currentVersion: "v1.20.0",
      keep: 3,
      keys,
      minimumSupportedVersion: "v1.16.0",
    });

    expect(plan.deleteKeys).toEqual([
      "lime/stable/v1.17.0/win32-x64/Lime-1.17.0 Setup.exe",
    ]);
    expect(plan.protectedVersions).toContain("1.16.0");
  });
});

describe("GitHub release asset staging", () => {
  it("拒绝旧 updater 资产进入 GitHub Release 资产准备", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-github-release-legacy-"),
    );
    const assetsDir = path.join(root, "release-assets");
    const outDir = path.join(root, "release-github-assets");

    writeFile(
      path.join(assetsDir, "aarch64-apple-darwin", "Lime_1.20.0_aarch64.dmg"),
    );
    writeFile(path.join(assetsDir, "aarch64-apple-darwin", "latest.json"));

    expect(() =>
      prepareGitHubReleaseAssets({
        assetsDir,
        outDir,
      }),
    ).toThrow(/legacy updater assets are not allowed/);
  });

  it("同名 Forge / Squirrel metadata 上传 GitHub Release 前应按平台重命名", () => {
    const root = fs.mkdtempSync(
      path.join(os.tmpdir(), "lime-github-release-assets-"),
    );
    const assetsDir = path.join(root, "release-assets");
    const outDir = path.join(root, "release-github-assets");

    writeFile(
      path.join(assetsDir, "aarch64-apple-darwin", "RELEASES.json"),
      "arm-feed",
    );
    writeFile(path.join(assetsDir, "aarch64-apple-darwin", "Lime.dmg"));
    writeFile(
      path.join(assetsDir, "x86_64-apple-darwin", "RELEASES.json"),
      "x64-feed",
    );
    writeFile(path.join(assetsDir, "x86_64-apple-darwin", "Lime.dmg"));
    writeFile(
      path.join(assetsDir, "x86_64-pc-windows-msvc", "Lime-1.29.0 Setup.exe"),
    );
    writeFile(path.join(assetsDir, "x86_64-pc-windows-msvc", "RELEASES"));

    const copied = prepareGitHubReleaseAssets({
      assetsDir,
      outDir,
      version: "v1.29.0",
    });

    expect(copied.map((item) => item.name).sort()).toEqual(
      [
        "Lime-1.29.0 Setup.exe",
        "macos-arm64-Lime-1.29.0.dmg",
        "macos-arm64-RELEASES.json",
        "macos-x64-Lime-1.29.0.dmg",
        "macos-x64-RELEASES.json",
        "RELEASES",
      ].sort(),
    );
    expect(
      fs.readFileSync(path.join(outDir, "macos-arm64-RELEASES.json"), "utf8"),
    ).toBe("arm-feed");
    expect(
      fs.readFileSync(path.join(outDir, "macos-x64-RELEASES.json"), "utf8"),
    ).toBe("x64-feed");
  });
});
